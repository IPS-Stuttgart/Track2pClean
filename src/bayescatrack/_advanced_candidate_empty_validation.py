"""Handle empty advanced candidate-pruning cost matrices safely.

Advanced candidate pruning is also used in diagnostic and sparse-data paths where
one side of an ROI association can be empty.  The base helper already produces an
empty admissibility mask before optional margin gating, but margin gating then
reduces along empty axes and raises a low-level NumPy error.  This patch preserves
normal validation while returning the mathematically empty candidate mask for
``(0, n)`` and ``(n, 0)`` matrices.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_advanced_candidate_empty_validation_patch"
_ALLOWED_KWARGS = frozenset({"top_k", "include_columns", "gate_margin", "large_cost"})


def install_advanced_candidate_empty_validation() -> None:
    """Install an idempotent empty-matrix guard for candidate pruning."""

    from . import (
        advanced_roi_components as advanced,  # pylint: disable=import-outside-toplevel
    )

    original_candidate_mask = advanced.candidate_mask_from_cost_matrix
    if getattr(original_candidate_mask, _PATCH_MARKER, False):
        return

    @wraps(original_candidate_mask)
    def candidate_mask_from_cost_matrix_with_empty_guard(
        cost_matrix: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if args or "top_k" not in kwargs or not set(kwargs).issubset(_ALLOWED_KWARGS):
            return original_candidate_mask(cost_matrix, *args, **kwargs)

        costs = np.asarray(cost_matrix, dtype=float)
        if costs.ndim != 2 or 0 not in costs.shape:
            return original_candidate_mask(cost_matrix, *args, **kwargs)

        _ = advanced._normalize_optional_positive_int(  # pylint: disable=protected-access
            kwargs["top_k"],
            name="top_k",
        )
        _ = advanced._normalize_bool(  # pylint: disable=protected-access
            kwargs.get("include_columns", True),
            name="include_columns",
        )
        _ = advanced._normalize_optional_nonnegative_float(  # pylint: disable=protected-access
            kwargs.get("gate_margin", None),
            name="gate_margin",
        )
        large_cost = (
            advanced._normalize_positive_float(  # pylint: disable=protected-access
                kwargs.get("large_cost", 1.0e6),
                name="large_cost",
            )
        )
        return np.isfinite(costs) & (costs < large_cost)

    setattr(candidate_mask_from_cost_matrix_with_empty_guard, _PATCH_MARKER, True)
    setattr(
        candidate_mask_from_cost_matrix_with_empty_guard,
        "_bayescatrack_original",
        original_candidate_mask,
    )
    advanced.candidate_mask_from_cost_matrix = (
        candidate_mask_from_cost_matrix_with_empty_guard
    )


__all__ = ["install_advanced_candidate_empty_validation"]
