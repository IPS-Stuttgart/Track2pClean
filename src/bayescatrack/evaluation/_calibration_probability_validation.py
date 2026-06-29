"""Probability-input validation for calibration diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

_ERROR_MESSAGE = "probabilities must be numeric, not text"


def checked_probability_vector(values: Any, original: Any) -> np.ndarray:
    """Return a probability vector after rejecting text-like scalars and arrays."""

    try:
        raw_values = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be numeric") from exc
    if _contains_text(raw_values):
        raise ValueError(_ERROR_MESSAGE)
    return original(values)


def _contains_text(values: np.ndarray) -> bool:
    return any(
        isinstance(value, (str, bytes, bytearray, np.str_, np.bytes_))
        for value in values.reshape(-1)
    )


__all__ = ["checked_probability_vector"]
