"""Binary-label helpers for calibration diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

ERROR_MESSAGE = "labels must be binary values 0/1 or False/True"


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


__all__ = ["as_binary_label_vector"]
