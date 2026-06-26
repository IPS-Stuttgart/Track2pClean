"""Calibration metrics for probabilistic association models."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ("brier_score",)


def brier_score(
    probabilities: Any, labels: Any, *, sample_weight: Any | None = None
) -> float:
    """Return the binary Brier score for calibrated match probabilities.

    The Brier score is the mean squared error between predicted probabilities
    and binary labels. Lower is better; a perfectly calibrated and perfectly
    accurate probabilistic classifier has score 0.
    """

    probabilities = _as_float_vector(
        probabilities,
        name="probabilities",
        reject_bool=True,
    )
    labels = _as_float_vector(labels, name="labels", reject_bool=False)

    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same flattened shape")
    if probabilities.size == 0:
        raise ValueError("At least one probability/label pair is required")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if not np.all(np.isfinite(labels)):
        raise ValueError("labels must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("labels must be binary 0/1 values")

    squared_errors = np.square(probabilities - labels)
    if sample_weight is None:
        return float(np.mean(squared_errors))

    weights = _validate_sample_weight(sample_weight, expected_shape=labels.shape)
    return float(np.average(squared_errors, weights=weights))


def _validate_sample_weight(
    sample_weight: Any, *, expected_shape: tuple[int, ...]
) -> np.ndarray:
    weights = _as_float_vector(
        sample_weight,
        name="sample_weight",
        reject_bool=True,
    )
    if weights.shape != expected_shape:
        raise ValueError("sample_weight must match the flattened label shape")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must be finite")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("At least one sample weight must be positive")
    return weights


def _as_float_vector(values: Any, *, name: str, reject_bool: bool) -> np.ndarray:
    try:
        raw_values = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if reject_bool and any(
        isinstance(value, (bool, np.bool_)) for value in raw_values.flat
    ):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        return np.asarray(raw_values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
