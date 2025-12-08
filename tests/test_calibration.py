"""Tests for calibration functionality."""

import pytest
import numpy as np
import pandas as pd


class TestCalibrator:
    """Test calibration of LLM responses to GSS ground truth."""

    def test_calibrator_initialization(self) -> None:
        """Calibrator can be initialized."""
        from hivesight_calibration.calibration import Calibrator

        calibrator = Calibrator()
        assert calibrator is not None

    def test_fit_learns_correction(self) -> None:
        """fit() learns correction from LLM vs actual responses."""
        from hivesight_calibration.calibration import Calibrator

        # Simulated data: LLM tends to be more extreme
        data = pd.DataFrame({
            "llm_response": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10,
            "actual_response": [1, 2, 2, 2, 3, 3, 3, 4, 4, 5] * 10,
            "age": [30] * 100,
            "income": [50000] * 100,
        })

        calibrator = Calibrator()
        calibrator.fit(
            llm_responses=data["llm_response"],
            actual_responses=data["actual_response"],
            features=data[["age", "income"]],
        )

        assert calibrator.is_fitted

    def test_predict_returns_distribution(self) -> None:
        """predict() returns probability distribution over responses."""
        from hivesight_calibration.calibration import Calibrator

        # Create and fit calibrator with mock data
        data = pd.DataFrame({
            "llm_response": [1, 2, 3, 4, 5] * 20,
            "actual_response": [1, 2, 3, 4, 5] * 20,
            "age": [30, 40, 50, 60, 70] * 20,
            "income": [40000, 60000, 80000, 100000, 50000] * 20,
        })

        calibrator = Calibrator()
        calibrator.fit(
            llm_responses=data["llm_response"],
            actual_responses=data["actual_response"],
            features=data[["age", "income"]],
        )

        # Predict for new LLM response
        probs = calibrator.predict(
            llm_response=3,
            features=pd.DataFrame({"age": [45], "income": [70000]}),
        )

        # Should return probabilities for each response category
        assert len(probs) == 5  # 5-point scale
        assert abs(sum(probs) - 1.0) < 0.001  # Sums to 1
        assert all(p >= 0 for p in probs)  # All non-negative


class TestCalibrationResult:
    """Test calibration result container."""

    def test_result_stores_metrics(self) -> None:
        """CalibrationResult stores evaluation metrics."""
        from hivesight_calibration.calibration import CalibrationResult

        result = CalibrationResult(
            crps=0.15,
            pinball_losses={0.1: 0.08, 0.5: 0.12, 0.9: 0.09},
            coverage={0.5: 0.48, 0.9: 0.88},
            n_train=1000,
            n_test=200,
        )

        assert result.crps == 0.15
        assert result.coverage[0.9] == 0.88

    def test_crps_calculation(self) -> None:
        """CRPS is calculated correctly."""
        from hivesight_calibration.calibration import calculate_crps

        # Perfect prediction: all mass on correct answer
        predicted_probs = np.array([0, 0, 1, 0, 0])  # Predicts category 3
        actual = 3  # Actual is category 3 (0-indexed: 2)

        crps = calculate_crps(predicted_probs, actual - 1)  # Convert to 0-indexed
        assert crps < 0.01  # Near-perfect

        # Bad prediction: mass on wrong answer
        predicted_probs = np.array([1, 0, 0, 0, 0])  # Predicts category 1
        actual = 5  # Actual is category 5

        crps = calculate_crps(predicted_probs, actual - 1)
        assert crps > 0.5  # Should be high (bad)

    def test_pinball_loss_calculation(self) -> None:
        """Pinball loss is calculated correctly."""
        from hivesight_calibration.calibration import calculate_pinball_loss

        # Median prediction matches actual
        predicted_quantile = 3.0
        actual = 3.0
        tau = 0.5

        loss = calculate_pinball_loss(predicted_quantile, actual, tau)
        assert loss == 0.0

        # Under-prediction
        predicted_quantile = 2.0
        actual = 4.0
        tau = 0.5

        loss = calculate_pinball_loss(predicted_quantile, actual, tau)
        assert loss == 1.0  # (4-2) * 0.5


class TestCrossValidation:
    """Test cross-validation by question."""

    def test_split_by_question(self) -> None:
        """Data can be split by question for proper evaluation."""
        from hivesight_calibration.calibration import split_by_question

        data = pd.DataFrame({
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "respondent_id": [1, 2, 1, 2, 1, 2],
            "response": [1, 2, 3, 4, 5, 1],
        })

        train, test = split_by_question(
            data, test_questions=["q3"], question_col="question_id"
        )

        assert len(train) == 4  # q1 and q2
        assert len(test) == 2  # q3
        assert set(test["question_id"]) == {"q3"}
        assert "q3" not in set(train["question_id"])
