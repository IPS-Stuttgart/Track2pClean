"""Strict validation for global-assignment track-row ROI entries.

The global tracker receives a matrix from the PyRecEst path-cover solver and then
normalizes missing entries before reporting Suite2p ROI indices.  The old coercion
used plain ``int(...)`` on every non-missing value, which can silently turn
booleans, numeric strings, or fractional floats into valid-looking ROI indices.
These hooks reject malformed ROI entries before global tracking summaries and
exports can depend on a fabricated Suite2p ROI id.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_global_track_row_validation_patch"
_ERROR_MESSAGE = (
    "global assignment track matrix must contain non-negative integer ROI indices "
    "or missing values"
)


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
        _validate_global_track_rows(track_rows)
        return original(track_rows, fill_value=fill_value)

    setattr(coerce_global_track_rows_with_roi_validation, _PATCH_MARKER, True)
    setattr(
        coerce_global_track_rows_with_roi_validation, "_bayescatrack_original", original
    )
    _tracking._coerce_global_track_rows = (
        coerce_global_track_rows_with_roi_validation  # pylint: disable=protected-access
    )


def _validate_global_track_rows(track_rows: Any) -> None:
    try:
        track_array = np.asarray(track_rows, dtype=object)
    except ValueError:
        return
    if track_array.ndim != 2:
        return

    for _, value in np.ndenumerate(track_array):
        _validate_global_track_value(value)


def _validate_global_track_value(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_ERROR_MESSAGE)
        return

    try:
        operator.index(value)
    except TypeError as exc:
        raise ValueError(_ERROR_MESSAGE) from exc


__all__ = ["install_global_track_row_validation"]
