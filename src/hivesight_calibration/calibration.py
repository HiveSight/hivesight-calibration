"""Calibration of LLM responses to ground truth survey data."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class CalibrationResult:
    """Results from calibration evaluation."""

    crps: float
    pinball_losses: dict[float, float]
    coverage: dict[float, float]
    n_train: int
    n_test: int
    model_params: dict[str, Any] = field(default_factory=dict)


def calculate_crps(predicted_probs: np.ndarray, actual_idx: int) -> float:
    """Calculate Continuous Ranked Probability Score.

    CRPS measures the quality of probabilistic predictions.
    Lower is better (0 is perfect).

    Args:
        predicted_probs: Probability distribution over categories.
        actual_idx: Index of the actual observed category (0-indexed).

    Returns:
        CRPS value.
    """
    n_categories = len(predicted_probs)
    cumulative_pred = np.cumsum(predicted_probs)

    # Create indicator for actual value
    actual_indicator = np.zeros(n_categories)
    actual_indicator[actual_idx:] = 1.0

    # CRPS = integral of (F(x) - I(x >= actual))^2
    crps = np.sum((cumulative_pred - actual_indicator) ** 2) / n_categories

    return float(crps)


def calculate_pinball_loss(predicted_quantile: float, actual: float, tau: float) -> float:
    """Calculate pinball loss for quantile prediction.

    Args:
        predicted_quantile: Predicted value at quantile tau.
        actual: Actual observed value.
        tau: Quantile level (0 to 1).

    Returns:
        Pinball loss value.
    """
    error = actual - predicted_quantile

    if error >= 0:
        return tau * error
    else:
        return (tau - 1) * error


def split_by_question(
    data: pd.DataFrame,
    test_questions: list[str],
    question_col: str = "question_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by question for proper evaluation.

    This ensures we test generalization to new questions,
    not just new respondents for seen questions.

    Args:
        data: Full dataset with question identifiers.
        test_questions: List of question IDs to hold out for testing.
        question_col: Column name containing question identifiers.

    Returns:
        Tuple of (train_data, test_data).
    """
    train_mask = ~data[question_col].isin(test_questions)
    test_mask = data[question_col].isin(test_questions)

    return data[train_mask].copy(), data[test_mask].copy()


class Calibrator:
    """Calibrate LLM responses to match true survey distributions."""

    def __init__(self, n_categories: int = 5) -> None:
        """Initialize the calibrator.

        Args:
            n_categories: Number of response categories (e.g., 5 for Likert).
        """
        self.n_categories = n_categories
        self.is_fitted = False
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None

    def fit(
        self,
        llm_responses: pd.Series,
        actual_responses: pd.Series,
        features: pd.DataFrame,
    ) -> None:
        """Fit the calibration model.

        Learns P(actual_response | llm_response, demographics).

        Args:
            llm_responses: LLM predicted responses (1-5 scale).
            actual_responses: Ground truth responses (1-5 scale).
            features: Demographic features (age, income, etc.).
        """
        # Prepare features: combine LLM response with demographics
        X = features.copy()
        X["llm_response"] = llm_responses.values

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Target: actual responses (0-indexed for sklearn)
        y = actual_responses.values - 1  # Convert to 0-indexed

        # Fit multinomial logistic regression
        self._model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self._model.fit(X_scaled, y)

        self.is_fitted = True

    def predict(
        self,
        llm_response: int,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Predict calibrated probability distribution.

        Args:
            llm_response: LLM's predicted response (1-5 scale).
            features: Demographic features for the prediction.

        Returns:
            Array of probabilities for each response category.
        """
        if not self.is_fitted or self._model is None or self._scaler is None:
            raise RuntimeError("Calibrator must be fit before predicting")

        # Prepare features
        X = features.copy()
        X["llm_response"] = llm_response

        # Scale
        X_scaled = self._scaler.transform(X)

        # Predict probabilities
        probs = self._model.predict_proba(X_scaled)[0]

        # Ensure we have probabilities for all categories
        # (sklearn may not include categories not seen in training)
        full_probs = np.zeros(self.n_categories)
        for i, class_label in enumerate(self._model.classes_):
            full_probs[class_label] = probs[i]

        # Normalize (in case of missing categories)
        if full_probs.sum() > 0:
            full_probs = full_probs / full_probs.sum()

        return full_probs

    def evaluate(
        self,
        llm_responses: pd.Series,
        actual_responses: pd.Series,
        features: pd.DataFrame,
        quantiles: list[float] | None = None,
    ) -> CalibrationResult:
        """Evaluate calibration on held-out data.

        Args:
            llm_responses: LLM predictions.
            actual_responses: Ground truth.
            features: Demographic features.
            quantiles: Quantile levels for pinball loss.

        Returns:
            CalibrationResult with metrics.
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        crps_values: list[float] = []
        pinball_losses: dict[float, list[float]] = {q: [] for q in quantiles}
        coverage_counts: dict[float, int] = {q: 0 for q in quantiles}

        n_samples = len(llm_responses)

        for i in range(n_samples):
            llm_resp = llm_responses.iloc[i]
            actual = actual_responses.iloc[i]
            feats = features.iloc[[i]]

            # Get calibrated distribution
            probs = self.predict(llm_resp, feats)

            # CRPS
            actual_idx = int(actual) - 1  # Convert to 0-indexed
            crps_values.append(calculate_crps(probs, actual_idx))

            # Pinball loss and coverage for each quantile
            cumulative = np.cumsum(probs)
            for q in quantiles:
                # Find predicted quantile value
                pred_quantile = np.searchsorted(cumulative, q) + 1  # 1-indexed

                # Pinball loss
                pl = calculate_pinball_loss(pred_quantile, actual, q)
                pinball_losses[q].append(pl)

        # Aggregate metrics
        avg_crps = float(np.mean(crps_values))
        avg_pinball = {q: float(np.mean(losses)) for q, losses in pinball_losses.items()}

        # Calculate coverage
        coverage: dict[float, float] = {}
        for q in quantiles:
            # For interval coverage, check if actual falls within prediction interval
            # This is a simplified version
            coverage[q] = 0.0  # TODO: Implement proper interval coverage

        return CalibrationResult(
            crps=avg_crps,
            pinball_losses=avg_pinball,
            coverage=coverage,
            n_train=0,  # Would need to track from fit
            n_test=n_samples,
        )
