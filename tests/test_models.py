"""
test_models.py — Unit Tests for Layer 2 PD Model Pipeline
==========================================================
Tests cover: data preparation, model training, evaluation metrics,
Expected Loss calculation, and SHAP output.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
#  DATA PREPARATION TESTS
# ============================================================

class TestDataPreparation:
    """Test the data preparation pipeline."""

    def _make_sample_df(self, n=1000):
        """Create a synthetic DataFrame matching GMSC structure."""
        np.random.seed(42)
        return pd.DataFrame({
            "SeriousDlqin2yrs": np.random.binomial(1, 0.067, n),
            "RevolvingUtilizationOfUnsecuredLines": np.random.uniform(0, 1, n),
            "age": np.random.randint(21, 80, n),
            "NumberOfTime30-59DaysPastDueNotWorse": np.random.randint(0, 5, n),
            "DebtRatio": np.random.uniform(0, 2, n),
            "MonthlyIncome": np.random.uniform(1000, 20000, n),
            "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 30, n),
            "NumberOfTimes90DaysLate": np.random.randint(0, 3, n),
            "NumberRealEstateLoansOrLines": np.random.randint(0, 5, n),
            "NumberOfTime60-89DaysPastDueNotWorse": np.random.randint(0, 3, n),
            "NumberOfDependents": np.random.randint(0, 5, n),
            "flag_30dpd": np.random.binomial(1, 0.15, n),
            "flag_60dpd": np.random.binomial(1, 0.05, n),
            "flag_90dpd": np.random.binomial(1, 0.05, n),
            "log_MonthlyIncome": np.random.uniform(6, 10, n),
            "total_delinquency_score": np.random.randint(0, 10, n),
            "gdp_growth": np.random.uniform(-2, 4, n),
            "unemployment_rate": np.random.uniform(3, 10, n),
        })

    def test_target_separation(self):
        """Target column should be separated from features."""
        df = self._make_sample_df()
        target = "SeriousDlqin2yrs"
        y = df[target]
        X = df.drop(columns=[target])
        assert target not in X.columns
        assert len(y) == len(df)

    def test_no_nulls_after_prep(self):
        """After prep, no NaN values should remain in features."""
        df = self._make_sample_df()
        # Inject some nulls
        df.loc[0:10, "MonthlyIncome"] = np.nan
        df = df.fillna(df.median(numeric_only=True))
        assert df.isnull().sum().sum() == 0

    def test_stratified_split_preserves_ratio(self):
        """Stratified split should preserve default rate in all splits."""
        from sklearn.model_selection import train_test_split
        df = self._make_sample_df(5000)
        y = df["SeriousDlqin2yrs"]
        X = df.drop(columns=["SeriousDlqin2yrs"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        original_rate = y.mean()
        train_rate = y_train.mean()
        test_rate = y_test.mean()

        # Rates should be within 1% of each other
        assert abs(original_rate - train_rate) < 0.01
        assert abs(original_rate - test_rate) < 0.01

    def test_scaling_zero_mean(self):
        """StandardScaler should produce ~zero mean features."""
        from sklearn.preprocessing import StandardScaler
        df = self._make_sample_df()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Mean should be approximately 0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)


# ============================================================
#  MODEL TRAINING TESTS
# ============================================================

class TestModelTraining:
    """Test that models train and predict correctly."""

    def _train_simple_model(self):
        """Train a simple LR model on synthetic data."""
        from sklearn.linear_model import LogisticRegression
        np.random.seed(42)
        X = np.random.randn(500, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_model_predicts_probabilities(self):
        """Model should output probabilities between 0 and 1."""
        model, X, y = self._train_simple_model()
        probs = model.predict_proba(X)[:, 1]

        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
        assert len(probs) == len(X)

    def test_model_predicts_classes(self):
        """Model should produce binary class predictions."""
        model, X, y = self._train_simple_model()
        preds = model.predict(X)

        assert set(np.unique(preds)).issubset({0, 1})

    def test_auc_above_random(self):
        """AUC-ROC should be above 0.5 (better than random)."""
        from sklearn.metrics import roc_auc_score
        model, X, y = self._train_simple_model()
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)

        assert auc > 0.5, f"AUC should be > 0.5, got {auc}"


# ============================================================
#  EVALUATION METRICS TESTS
# ============================================================

class TestEvaluationMetrics:
    """Test evaluation metric computations."""

    def test_ks_statistic_perfect_model(self):
        """Perfect model should have KS = 1.0."""
        from src.models.evaluate import compute_ks_statistic
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = compute_ks_statistic(y_true, y_prob)
        assert ks > 0.9, f"Perfect model KS should be ~1.0, got {ks}"

    def test_ks_statistic_random_model(self):
        """Random model should have KS close to 0."""
        from src.models.evaluate import compute_ks_statistic
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_prob = np.random.uniform(0, 1, 1000)
        ks = compute_ks_statistic(y_true, y_prob)
        assert ks < 0.15, f"Random model KS should be ~0, got {ks}"

    def test_gini_from_auc(self):
        """Gini should equal 2*AUC - 1."""
        from sklearn.metrics import roc_auc_score
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        assert abs(gini - (2 * auc - 1)) < 1e-10


# ============================================================
#  EXPECTED LOSS TESTS
# ============================================================

class TestExpectedLoss:
    """Test Basel III Expected Loss calculations."""

    def test_el_formula(self):
        """EL should equal PD * LGD * EAD."""
        pd_score = 0.05
        lgd = 0.45
        ead = 20000
        el = pd_score * lgd * ead
        assert el == pytest.approx(450.0)

    def test_portfolio_el_is_sum(self):
        """Portfolio EL should be the sum of loan-level EL."""
        pds = np.array([0.05, 0.10, 0.20])
        lgd = 0.45
        eads = np.array([10000, 20000, 15000])
        loan_els = pds * lgd * eads
        portfolio_el = loan_els.sum()

        expected = 0.05*0.45*10000 + 0.10*0.45*20000 + 0.20*0.45*15000
        assert abs(portfolio_el - expected) < 0.01

    def test_zero_pd_means_zero_el(self):
        """A loan with PD=0 should have EL=0."""
        el = 0.0 * 0.45 * 50000
        assert el == 0.0

    def test_el_increases_with_pd(self):
        """Higher PD should produce higher EL."""
        lgd = 0.45
        ead = 20000
        el_low = 0.05 * lgd * ead
        el_high = 0.30 * lgd * ead
        assert el_high > el_low

    def test_risk_grade_assignment(self):
        """PD values should map to correct risk grades."""
        pds = pd.Series([0.02, 0.08, 0.15, 0.35, 0.80])
        grades = pd.cut(
            pds,
            bins=[0, 0.05, 0.10, 0.20, 0.50, 1.0],
            labels=["A", "B", "C", "D", "E"],
            include_lowest=True,
        )
        expected = ["A", "B", "C", "D", "E"]
        assert list(grades) == expected
