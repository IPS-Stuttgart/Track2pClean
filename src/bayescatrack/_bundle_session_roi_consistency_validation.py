"""Validate ROI-index consistency across consecutive association bundles.

When track rows are seeded from an intermediate session, that session is described
by the previous bundle's measurement ROI indices and by the next bundle's
reference ROI indices. Those two views must agree exactly; otherwise track-row
construction can silently seed tracks from a different ROI layout than the one
used by one side of the stitched matches.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_bundle_session_roi_consistency_validation_patch"


def install_bundle_session_roi_consistency_validation(matching_module: Any) -> None:
    """Install an idempotent guard for intermediate-session ROI layouts."""

    original_bundle_roi_indices_for_session = matching_module._bundle_roi_indices_for_session
    if getattr(original_bundle_roi_indices_for_session, _PATCH_MARKER, False):
        return

    @wraps(original_bundle_roi_indices_for_session)
    def _bundle_roi_indices_for_session_with_validation(
        bundles: Sequence[Any],
        session_index: int,
    ) -> np.ndarray:
        n_bundles = len(bundles)
        normalized_session_index = matching_module._normalize_session_index(
            session_index,
            "start_session_index",
            num_sessions=n_bundles + 1,
        )
        if 0 < normalized_session_index < n_bundles:
            previous_measurement_indices = _normalized_bundle_roi_indices(
                matching_module,
                bundles[normalized_session_index - 1],
                "measurement_roi_indices",
                axis=1,
            )
            next_reference_indices = _normalized_bundle_roi_indices(
                matching_module,
                bundles[normalized_session_index],
                "reference_roi_indices",
                axis=0,
            )
            if not np.array_equal(previous_measurement_indices, next_reference_indices):
                raise ValueError(
                    "inconsistent ROI indices for intermediate session "
                    f"{normalized_session_index}: previous bundle measurement_roi_indices "
                    "must match next bundle reference_roi_indices"
                )
            return previous_measurement_indices

        return original_bundle_roi_indices_for_session(bundles, session_index)

    setattr(_bundle_roi_indices_for_session_with_validation, _PATCH_MARKER, True)
    setattr(
        _bundle_roi_indices_for_session_with_validation,
        "_bayescatrack_original",
        original_bundle_roi_indices_for_session,
    )
    matching_module._bundle_roi_indices_for_session = (
        _bundle_roi_indices_for_session_with_validation
    )


def _normalized_bundle_roi_indices(
    matching_module: Any,
    bundle: Any,
    field_name: str,
    *,
    axis: int,
) -> np.ndarray:
    cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("bundle.pairwise_cost_matrix must be two-dimensional")
    return matching_module._normalize_bundle_roi_indices(
        getattr(bundle, field_name),
        field_name,
        expected_length=cost_matrix.shape[axis],
    )


__all__ = ["install_bundle_session_roi_consistency_validation"]
