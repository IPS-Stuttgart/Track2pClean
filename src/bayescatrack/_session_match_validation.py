"""Strict validation for :class:`bayescatrack.matching.SessionMatchResult`."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


def install_session_match_result_validation() -> None:
    """Install an idempotent validation hook for session-match result arrays."""

    from . import matching as _matching

    cls = _matching.SessionMatchResult
    if getattr(cls, "_bayescatrack_session_match_result_validation_patch", False):
        return

    original_post_init = cls.__post_init__

    def _validated_post_init(self: Any) -> None:
        original_post_init(self)
        for field_name in (
            "reference_positions",
            "measurement_positions",
            "reference_roi_indices",
            "measurement_roi_indices",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer_array(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "costs",
            _finite_cost_array(getattr(self, "costs"), "costs"),
        )
        _reject_duplicate_roi_indices(
            getattr(self, "reference_roi_indices"),
            "reference_roi_indices",
        )
        _reject_duplicate_roi_indices(
            getattr(self, "measurement_roi_indices"),
            "measurement_roi_indices",
        )

    setattr(
        _validated_post_init, "_bayescatrack_original_post_init", original_post_init
    )
    setattr(
        _validated_post_init,
        "_bayescatrack_session_match_result_validation_patch",
        True,
    )
    cls.__post_init__ = _validated_post_init
    setattr(cls, "_bayescatrack_session_match_result_validation_patch", True)


def _nonnegative_integer_array(values: Any, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    normalized = np.empty(array.shape, dtype=int)
    for index, value in np.ndenumerate(array):
        try:
            normalized[index] = _nonnegative_integer(value, field_name)
        except OverflowError as exc:
            raise ValueError(
                f"{field_name} must contain non-negative integer values"
            ) from exc
    return normalized


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain non-negative integer values")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain non-negative integer values")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field_name} must contain non-negative integer values"
            ) from exc
    integer_value = int(integer_value)
    if integer_value < 0:
        raise ValueError(f"{field_name} must contain non-negative integer values")
    return integer_value


def _finite_cost_array(values: Any, field_name: str) -> np.ndarray:
    raw_array = np.asarray(values, dtype=object)
    if raw_array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if any(isinstance(value, (bool, np.bool_)) for value in raw_array.flat):
        raise ValueError(f"{field_name} must contain finite numeric assignment costs")
    try:
        array = np.asarray(raw_array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field_name} must contain finite numeric assignment costs"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain finite numeric assignment costs")
    return array


def _reject_duplicate_roi_indices(values: Any, field_name: str) -> None:
    array = np.asarray(values, dtype=int)
    seen: set[int] = set()
    duplicates: list[int] = []
    for value in array.tolist():
        roi_index = int(value)
        if roi_index in seen and roi_index not in duplicates:
            duplicates.append(roi_index)
        seen.add(roi_index)
    if duplicates:
        duplicate_summary = ", ".join(str(value) for value in duplicates)
        raise ValueError(
            f"{field_name} must contain unique ROI indices; "
            f"duplicate values: {duplicate_summary}"
        )


__all__ = ["install_session_match_result_validation"]
