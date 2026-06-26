"""Strict validation for linear-assignment bundle ROI-index vectors.

The assignment solver maps Hungarian row/column positions back to Suite2p ROI
indices using ``bundle.reference_roi_indices`` and
``bundle.measurement_roi_indices``.  The original implementation trusted these
arrays and coerced them with ``np.asarray(..., dtype=int)`` after solving.  That
can turn malformed values such as booleans or fractional floats into different
ROI IDs, and inconsistent vector lengths either surface raw NumPy ``IndexError``
exceptions or silently ignore extra ROI IDs.  This hook fails fast at the bundle
boundary before a malformed association bundle can change benchmark outputs.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_assignment_bundle_validation_patch"


def install_assignment_bundle_validation() -> None:
    """Install idempotent validation around assignment bundle ROI vectors."""

    from . import matching as _matching  # pylint: disable=import-outside-toplevel

    original_solve = _matching.solve_bundle_linear_assignment
    if getattr(original_solve, _PATCH_MARKER, False):
        return

    @wraps(original_solve)
    def solve_bundle_linear_assignment_with_bundle_validation(
        bundle: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        _validate_assignment_bundle_roi_indices(bundle)
        return original_solve(bundle, *args, **kwargs)

    setattr(solve_bundle_linear_assignment_with_bundle_validation, _PATCH_MARKER, True)
    setattr(
        solve_bundle_linear_assignment_with_bundle_validation,
        "_bayescatrack_original",
        original_solve,
    )
    _matching.solve_bundle_linear_assignment = (
        solve_bundle_linear_assignment_with_bundle_validation
    )


def _validate_assignment_bundle_roi_indices(bundle: Any) -> None:
    cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        return

    reference_roi_indices = _normalize_roi_index_array(
        bundle.reference_roi_indices,
        "bundle.reference_roi_indices",
    )
    measurement_roi_indices = _normalize_roi_index_array(
        bundle.measurement_roi_indices,
        "bundle.measurement_roi_indices",
    )

    if reference_roi_indices.shape[0] != cost_matrix.shape[0]:
        raise ValueError(
            "bundle.reference_roi_indices length must match "
            "pairwise_cost_matrix rows"
        )
    if measurement_roi_indices.shape[0] != cost_matrix.shape[1]:
        raise ValueError(
            "bundle.measurement_roi_indices length must match "
            "pairwise_cost_matrix columns"
        )


def _normalize_roi_index_array(values: Any, field_name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a one-dimensional ROI-index array")

    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional ROI-index array")

    normalized = tuple(
        _normalize_roi_index(value, field_name) for value in array.tolist()
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique ROI indices")
    return np.asarray(normalized, dtype=int)


def _normalize_roi_index(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain integer ROI indices")

    try:
        integer_value = int(operator.index(value))
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(f"{field_name} must contain integer ROI indices") from None
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain integer ROI indices")
        integer_value = int(numeric_value)

    if integer_value < 0:
        raise ValueError(f"{field_name} must contain non-negative ROI indices")
    return integer_value


__all__ = ["install_assignment_bundle_validation"]
