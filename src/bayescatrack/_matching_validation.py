"""Strict validation for linear-assignment bundle layouts and options."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np


_PATCH_MARKER = "_bayescatrack_assignment_layout_validation_patch"


def install_matching_layout_validation(matching_module: Any) -> None:
    """Install an idempotent validator on matching assignment solves."""

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


__all__ = ["install_matching_layout_validation"]
