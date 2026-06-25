"""Strict validation for matching bundle ROI-index metadata.

The linear-assignment solver maps assignment row/column positions back to source
ROI indices.  The core implementation used NumPy integer coercion at that final
indexing step, so malformed bundle metadata could either be silently truncated
(for example fractional float ROI IDs) or fail as a low-level indexing error
when the ROI-index arrays did not match the cost-matrix dimensions.

This import-time patch preserves the public matching API while failing fast with
consistent ``ValueError`` diagnostics before assignment results are converted
back to ROI indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np

from . import matching as _matching

_PATCH_MARKER = "_bayescatrack_bundle_roi_index_validation_patch"


@dataclass(frozen=True)
class _BundleWithValidatedRoiIndices:
    original: Any
    reference_roi_indices: np.ndarray
    measurement_roi_indices: np.ndarray

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


def install_matching_bundle_roi_index_validation() -> None:
    """Install idempotent validation around matching bundle ROI metadata."""

    original = _matching.solve_bundle_linear_assignment
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def solve_bundle_linear_assignment_with_bundle_roi_index_validation(
        bundle: Any,
        *,
        max_cost: float | None = _matching.DEFAULT_ASSIGNMENT_MAX_COST,
    ) -> _matching.SessionMatchResult:
        cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
        if cost_matrix.ndim == 2:
            bundle = _BundleWithValidatedRoiIndices(
                original=bundle,
                reference_roi_indices=_normalize_bundle_roi_indices(
                    bundle.reference_roi_indices,
                    "reference_roi_indices",
                    expected_length=int(cost_matrix.shape[0]),
                    axis_name="row",
                ),
                measurement_roi_indices=_normalize_bundle_roi_indices(
                    bundle.measurement_roi_indices,
                    "measurement_roi_indices",
                    expected_length=int(cost_matrix.shape[1]),
                    axis_name="column",
                ),
            )
        return original(bundle, max_cost=max_cost)

    setattr(
        solve_bundle_linear_assignment_with_bundle_roi_index_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        solve_bundle_linear_assignment_with_bundle_roi_index_validation,
        "_bayescatrack_original",
        original,
    )
    _matching.solve_bundle_linear_assignment = (
        solve_bundle_linear_assignment_with_bundle_roi_index_validation
    )


def _normalize_bundle_roi_indices(
    values: Any,
    field_name: str,
    *,
    expected_length: int,
    axis_name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")

    normalized = _matching._normalize_roi_index_sequence(  # pylint: disable=protected-access
        array.tolist(),
        field_name,
        require_unique=True,
    )
    if len(normalized) != expected_length:
        raise ValueError(
            f"{field_name} length {len(normalized)} must match "
            f"pairwise_cost_matrix {axis_name} dimension {expected_length}"
        )
    return np.asarray(normalized, dtype=int)


__all__ = ["install_matching_bundle_roi_index_validation"]
