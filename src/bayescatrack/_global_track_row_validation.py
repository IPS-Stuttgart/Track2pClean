"""Strict validation for global-assignment track-row ROI entries.

The global tracker receives a matrix from the PyRecEst path-cover solver and then
normalizes missing entries before reporting Suite2p ROI indices.  The old coercion
used plain ``int(...)`` on every non-missing value, which can silently turn
booleans, numeric strings, fractional floats, or unconfigured negative sentinels
into valid-looking ROI indices or missing detections. These hooks reject
malformed ROI entries before global tracking summaries and exports can depend on
a fabricated Suite2p ROI id.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_global_track_row_validation_patch"
_ERROR_MESSAGE = (
    "global assignment track matrix must contain non-negative integer ROI indices "
    "or the configured missing fill_value"
)
_FILL_VALUE_ERROR = "fill_value must be a negative integer sentinel"


def install_global_track_row_validation() -> None:
    """Install idempotent validation around global tracking row coercion."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original = _tracking._coerce_global_track_rows  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def coerce_global_track_rows_with_roi_validation(
        track_rows: Any,
        *,
        fill_value: int,
    ) -> np.ndarray:
        normalized_fill_value = _normalize_fill_value(fill_value)
        _validate_global_track_rows(track_rows, fill_value=normalized_fill_value)
        return original(track_rows, fill_value=normalized_fill_value)

    setattr(coerce_global_track_rows_with_roi_validation, _PATCH_MARKER, True)
    setattr(coerce_global_track_rows_with_roi_validation, "_bayescatrack_original", original)
    _tracking._coerce_global_track_rows = coerce_global_track_rows_with_roi_validation  # pylint: disable=protected-access


def _validate_global_track_rows(track_rows: Any, *, fill_value: int) -> None:
    try:
        track_array = np.asarray(track_rows, dtype=object)
    except ValueError:
        return
    if track_array.ndim != 2:
        return

    for _, value in np.ndenumerate(track_array):
        _validate_global_track_value(value, fill_value=fill_value)


def _validate_global_track_value(value: Any, *, fill_value: int) -> None:
    if value is None:
        return

    integer_value = _normalize_integer_like(value, error_message=_ERROR_MESSAGE)
    if integer_value == fill_value:
        return
    if integer_value < 0:
        raise ValueError(_ERROR_MESSAGE)


def _normalize_fill_value(value: Any) -> int:
    integer_value = _normalize_integer_like(value, error_message=_FILL_VALUE_ERROR)
    if integer_value >= 0:
        raise ValueError(_FILL_VALUE_ERROR)
    return integer_value


def _normalize_integer_like(value: Any, *, error_message: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(error_message)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(error_message)
        return int(numeric_value)

    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(error_message) from exc


__all__ = ["install_global_track_row_validation"]
