"""Validate track-refinement track rows before platform-int casts.

Track-refinement helpers normalize row matrices to NumPy ``int`` arrays. Unsafely
casting unsigned or object integer arrays can wrap oversized ROI IDs into negative
values, after which they may look like missing-value sentinels.  This patch keeps
those malformed rows from reaching the cast.
"""

from __future__ import annotations

import operator
from decimal import Decimal
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_refinement_track_row_range_validation_patch"
_TRACK_ROW_ERROR = "track_rows must contain finite integer ROI indices"
_PLATFORM_INT_MIN = int(np.iinfo(np.int_).min)
_PLATFORM_INT_MAX = int(np.iinfo(np.int_).max)
_REJECTED_SCALARS = (bool, np.bool_, str, bytes, bytearray, np.bytes_)


def install_track_refinement_track_row_range_validation() -> None:
    """Install idempotent range validation for track-refinement row matrices."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original = module._validated_track_row_matrix
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _validated_track_row_matrix_with_range_validation(
        track_rows: Any,
    ) -> np.ndarray:
        _validate_track_row_integer_range(track_rows)
        return original(track_rows)

    setattr(_validated_track_row_matrix_with_range_validation, _PATCH_MARKER, True)
    setattr(
        _validated_track_row_matrix_with_range_validation,
        "_bayescatrack_original",
        original,
    )
    module._validated_track_row_matrix = (
        _validated_track_row_matrix_with_range_validation
    )


def _validate_track_row_integer_range(track_rows: Any) -> None:
    try:
        rows = np.asarray(track_rows, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_TRACK_ROW_ERROR) from exc

    if rows.ndim != 2:
        return

    for value in rows.ravel():
        integer_value = _track_row_integer_value(value)
        if integer_value < _PLATFORM_INT_MIN or integer_value > _PLATFORM_INT_MAX:
            raise ValueError(_TRACK_ROW_ERROR)


def _track_row_integer_value(value: Any) -> int:
    if isinstance(value, _REJECTED_SCALARS):
        raise ValueError(_TRACK_ROW_ERROR)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Decimal):
        return _integer_from_decimal(value)
    if isinstance(value, (float, np.floating)):
        return _integer_from_float(float(value))

    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        pass

    try:
        scalar_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_TRACK_ROW_ERROR) from exc
    if scalar_array.shape != ():
        raise ValueError(_TRACK_ROW_ERROR)

    scalar = scalar_array.item()
    if scalar is not value:
        return _track_row_integer_value(scalar)

    try:
        return _integer_from_float(float(scalar))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_TRACK_ROW_ERROR) from exc


def _integer_from_decimal(value: Decimal) -> int:
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(_TRACK_ROW_ERROR)
    return int(value)


def _integer_from_float(value: float) -> int:
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(_TRACK_ROW_ERROR)
    return int(value)


__all__ = ["install_track_refinement_track_row_range_validation"]
