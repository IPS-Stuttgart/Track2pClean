"""Binary-label helpers for calibration diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

ERROR_MESSAGE = "labels must be binary values 0/1 or False/True"


def checked_probability_label_arrays(
    probabilities: Any,
    labels: Any,
    to_probability_vector: Any,
) -> tuple[np.ndarray, np.ndarray]:
    probability_array = to_probability_vector(probabilities)
    label_array = as_binary_label_vector(labels)
    if probability_array.shape[0] == 0:
        raise ValueError("At least one probability is required")
    if probability_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            "probabilities and labels must contain the same number of entries"
        )
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must be finite")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return probability_array, label_array


def as_binary_label_vector(labels: Any) -> np.ndarray:
    try:
        raw_labels = np.asarray(labels, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(ERROR_MESSAGE) from exc
    normalized = np.empty(raw_labels.shape, dtype=float)
    for index, value in enumerate(raw_labels):
        normalized[index] = binary_label_value(value)
    return normalized


def binary_label_value(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
        if integer_value in (0, 1):
            return float(integer_value)
        raise ValueError(ERROR_MESSAGE)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value in (0.0, 1.0):
            return numeric_value
    raise ValueError(ERROR_MESSAGE)


__all__ = ["as_binary_label_vector", "checked_probability_label_arrays"]
