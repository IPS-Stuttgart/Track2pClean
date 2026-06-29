"""Strict validation for global-assignment track matrices."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_global_assignment_track_row_validation_patch"
_ERROR_MESSAGE = (
    "global assignment track matrix must contain integer ROI indices "
    "or the configured missing fill_value"
)
_FILL_VALUE_ERROR = "fill_value must be a negative integer sentinel"


def install_global_assignment_track_row_validation() -> None:
    """Install idempotent validation around global assignment track-row coercion."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original = _tracking._coerce_global_track_rows  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def coerce_global_track_rows_with_validation(
        track_rows: Any, *, fill_value: Any
    ) -> np.ndarray:
        normalized_fill_value = _normalize_fill_value(fill_value)
        return original(
            _normalize_global_track_rows(track_rows, fill_value=normalized_fill_value),
            fill_value=normalized_fill_value,
        )

    setattr(coerce_global_track_rows_with_validation, _PATCH_MARKER, True)
    setattr(
        coerce_global_track_rows_with_validation, "_bayescatrack_original", original
    )
    _tracking._coerce_global_track_rows = (
        coerce_global_track_rows_with_validation  # pylint: disable=protected-access
    )


def _normalize_global_track_rows(values: Any, *, fill_value: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 2:
        raise ValueError("global assignment track matrix must be two-dimensional")
    normalized = np.empty(array.shape, dtype=int)
    for index, value in np.ndenumerate(array):
        normalized[index] = _normalize_global_track_value(value, fill_value=fill_value)
    return normalized


def _normalize_global_track_value(value: Any, *, fill_value: int) -> int:
    if value is None:
        return fill_value
    try:
        integer_value = _normalize_integer_like(value)
    except ValueError as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if integer_value == fill_value:
        return fill_value
    if integer_value < 0:
        raise ValueError(_ERROR_MESSAGE)
    return integer_value


def _normalize_fill_value(value: Any) -> int:
    try:
        integer_value = _normalize_integer_like(value)
    except ValueError as exc:
        raise ValueError(_FILL_VALUE_ERROR) from exc
    if integer_value >= 0:
        raise ValueError(_FILL_VALUE_ERROR)
    return integer_value


def _normalize_integer_like(value: Any) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("value must be an integer")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("value must be an integer")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("value must be an integer")
        return int(numeric_value)
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("value must be an integer") from exc


__all__ = ["install_global_assignment_track_row_validation"]
