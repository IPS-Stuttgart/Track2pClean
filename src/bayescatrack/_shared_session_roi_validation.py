"""Validate shared-session ROI indices across consecutive association bundles."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_shared_session_roi_validation_patch"


def install_shared_session_roi_validation(matching_module: Any) -> None:
    """Install an idempotent guard for bundle-to-track stitching."""

    original = matching_module.build_track_rows_from_bundles
    if getattr(original, _PATCH_ATTR, False):
        return

    @wraps(original)
    def build_track_rows_from_bundles(
        bundles: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        bundle_list = list(bundles)
        if not args and bundle_list:
            matching_module._normalize_session_index(  # pylint: disable=protected-access
                kwargs.get("start_session_index", 0),
                "start_session_index",
                num_sessions=len(bundle_list) + 1,
            )
            _validate_shared_session_roi_indices(matching_module, bundle_list)
        return original(bundle_list, *args, **kwargs)

    setattr(build_track_rows_from_bundles, _PATCH_ATTR, True)
    setattr(build_track_rows_from_bundles, "_bayescatrack_original", original)
    matching_module.build_track_rows_from_bundles = build_track_rows_from_bundles


def _validate_shared_session_roi_indices(
    matching_module: Any,
    bundles: list[Any],
) -> None:
    for bundle_index in range(len(bundles) - 1):
        previous_bundle = bundles[bundle_index]
        next_bundle = bundles[bundle_index + 1]
        shared_session_name = str(previous_bundle.measurement_session_name)
        if shared_session_name != str(next_bundle.reference_session_name):
            continue

        previous_costs = np.asarray(previous_bundle.pairwise_cost_matrix)
        next_costs = np.asarray(next_bundle.pairwise_cost_matrix)
        if previous_costs.ndim != 2 or next_costs.ndim != 2:
            continue

        previous_indices = matching_module._normalize_bundle_roi_indices(  # pylint: disable=protected-access
            previous_bundle.measurement_roi_indices,
            "measurement_roi_indices",
            expected_length=previous_costs.shape[1],
        )
        next_indices = matching_module._normalize_bundle_roi_indices(  # pylint: disable=protected-access
            next_bundle.reference_roi_indices,
            "reference_roi_indices",
            expected_length=next_costs.shape[0],
        )
        if previous_indices.shape == next_indices.shape and np.array_equal(
            previous_indices,
            next_indices,
        ):
            continue

        raise ValueError(
            "inconsistent ROI indices for intermediate session "
            f"{bundle_index + 1}: consecutive bundles disagree on ROI indices "
            f"for shared session {shared_session_name!r}: bundle {bundle_index} "
            "measurement_roi_indices do not match bundle "
            f"{bundle_index + 1} reference_roi_indices"
        )


__all__ = ["install_shared_session_roi_validation"]
