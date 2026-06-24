"""Fix candidate-margin pruning for empty ROI-pair matrices."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import advanced_roi_components as _advanced_roi_components

_ORIGINAL_ATTR = "_bayescatrack_empty_candidate_margin_original"
_PATCH_MARKER = "_bayescatrack_empty_candidate_margin_fix"
_VALIDATION_SENTINEL_COSTS = np.zeros((1, 1), dtype=float)


def install_empty_candidate_gate_margin_fix() -> None:
    """Install an idempotent guard for empty margin-gated candidate masks."""

    if getattr(_advanced_roi_components, _PATCH_MARKER, False):
        return
    original = _advanced_roi_components.candidate_mask_from_cost_matrix
    setattr(candidate_mask_from_cost_matrix, _ORIGINAL_ATTR, original)
    _advanced_roi_components.candidate_mask_from_cost_matrix = (
        candidate_mask_from_cost_matrix
    )
    setattr(_advanced_roi_components, _PATCH_MARKER, True)


def candidate_mask_from_cost_matrix(
    cost_matrix: np.ndarray,
    *,
    top_k: int | None,
    include_columns: bool = True,
    gate_margin: float | None = None,
    large_cost: float = 1.0e6,
) -> np.ndarray:
    """Return a candidate mask, including empty shapes under margin gating."""

    costs = np.asarray(cost_matrix, dtype=float)
    original = _original_candidate_mask()
    if costs.ndim == 2 and gate_margin is not None and 0 in costs.shape:
        # Delegate scalar validation to the implementation that was installed
        # before this guard, then avoid NumPy reductions over empty axes.
        original(
            _VALIDATION_SENTINEL_COSTS,
            top_k=top_k,
            include_columns=include_columns,
            gate_margin=gate_margin,
            large_cost=large_cost,
        )
        return np.zeros(costs.shape, dtype=bool)
    return original(
        cost_matrix,
        top_k=top_k,
        include_columns=include_columns,
        gate_margin=gate_margin,
        large_cost=large_cost,
    )


def _original_candidate_mask() -> Callable[..., np.ndarray]:
    original = getattr(candidate_mask_from_cost_matrix, _ORIGINAL_ATTR, None)
    if original is None:
        raise RuntimeError("empty candidate-margin guard is not installed")
    return original


__all__ = [
    "candidate_mask_from_cost_matrix",
    "install_empty_candidate_gate_margin_fix",
]
