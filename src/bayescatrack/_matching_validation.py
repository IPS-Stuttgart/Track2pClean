"""Strict validation for linear-assignment bundle layouts and options."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Callable

import numpy as np


_PATCH_MARKER = "_bayescatrack_assignment_layout_validation_patch"
_FILL_VALUE_PATCH_MARKER = "_bayescatrack_matching_fill_value_validation_patch"
_FILL_VALUE_ERROR_MESSAGE = "fill_value must be a negative integer sentinel"


def install_matching_layout_validation(matching_module: Any) -> None:
    """Install idempotent validators on matching assignment and row-stitching helpers."""

    _patch_assignment_solver(matching_module)
    _patch_fill_value_keyword_function(matching_module, "build_track_rows_from_matches")
    _patch_fill_value_keyword_function(matching_module, "build_track_rows_from_bundles")


def _patch_assignment_solver(matching_module: Any) -> None:
    original: Callable[..., Any] = matching_module.solve_bundle_linear_assignment
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def solve_bundle_linear_assignment(bundle: Any, *args: Any, **kwargs: Any) -> Any:
        _validate_bundle_assignment_layout(bundle)
        if "max_cost" in kwargs:
            kwargs = dict(kwargs)
            kwargs["max_cost"] = _normalize_max_cost(kwargs["max_cost"])
        return original(bundle, *args, **kwargs)

    setattr(solve_bundle_linear_assignment, _PATCH_MARKER, True)
    setattr(solve_bundle_linear_assignment, "_bayescatrack_original", original)
    matching_module.solve_bundle_linear_assignment = solve_bundle_linear_assignment


def _patch_fill_value_keyword_function(matching_module: Any, name: str) -> None:
    original: Callable[..., Any] = getattr(matching_module, name)
    if getattr(original, _FILL_VALUE_PATCH_MARKER, False):
        return

    @wraps(original)
    def function_with_fill_value_validation(*args: Any, **kwargs: Any) -> Any:
        if "fill_value" in kwargs:
            kwargs = dict(kwargs)
            kwargs["fill_value"] = _normalize_fill_value(kwargs["fill_value"])
        return original(*args, **kwargs)

    setattr(function_with_fill_value_validation, _FILL_VALUE_PATCH_MARKER, True)
    setattr(function_with_fill_value_validation, "_bayescatrack_original", original)
    setattr(matching_module, name, function_with_fill_value_validation)


def _validate_bundle_assignment_layout(bundle: Any) -> None:
    try:
        cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    except (TypeError, ValueError):
        return
    if cost_matrix.ndim != 2:
        return

    _validate_roi_index_axis(
        "reference_roi_indices",
        bundle.reference_roi_indices,
        expected_len=int(cost_matrix.shape[0]),
        axis_name="row",
    )
    _validate_roi_index_axis(
        "measurement_roi_indices",
        bundle.measurement_roi_indices,
        expected_len=int(cost_matrix.shape[1]),
        axis_name="column",
    )


def _validate_roi_index_axis(
    field_name: str,
    values: Any,
    *,
    expected_len: int,
    axis_name: str,
) -> None:
    roi_indices = np.asarray(values)
    if roi_indices.ndim != 1:
        raise ValueError(f"bundle.{field_name} must be one-dimensional")

    actual_len = int(roi_indices.shape[0])
    if actual_len != int(expected_len):
        raise ValueError(
            f"bundle.{field_name} length ({actual_len}) must match "
            f"pairwise_cost_matrix {axis_name} dimension ({expected_len})"
        )


def _normalize_max_cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("max_cost must be a finite non-negative value")
    try:
        max_cost = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_cost must be a finite non-negative value") from exc
    if not np.isfinite(max_cost) or max_cost < 0.0:
        raise ValueError("max_cost must be a finite non-negative value")
    return float(max_cost)


def _normalize_fill_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_FILL_VALUE_ERROR_MESSAGE)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(_FILL_VALUE_ERROR_MESSAGE)
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(_FILL_VALUE_ERROR_MESSAGE) from exc

    integer_value = int(integer_value)
    if integer_value >= 0:
        raise ValueError(
            "fill_value must be a negative integer sentinel that cannot collide "
            "with non-negative ROI indices"
        )
    return integer_value


__all__ = ["install_matching_layout_validation"]
