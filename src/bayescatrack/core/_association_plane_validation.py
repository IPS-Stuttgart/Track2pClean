"""Validation for transformed measurement planes in association bundles."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_association_plane_validation_patch"


def install_association_plane_validation(bridge_impl: Any) -> None:
    """Install idempotent validation on association-bundle construction."""

    original = bridge_impl.build_session_pair_association_bundle
    if _function_chain_has_patch(original):
        return

    @wraps(original)
    def build_session_pair_association_bundle_with_plane_validation(
        reference_session: Any,
        measurement_session: Any,
        *args: Any,
        measurement_plane_in_reference_frame: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        _validate_measurement_plane_replacement(
            measurement_session,
            measurement_plane_in_reference_frame,
        )
        return original(
            reference_session,
            measurement_session,
            *args,
            measurement_plane_in_reference_frame=measurement_plane_in_reference_frame,
            **kwargs,
        )

    setattr(build_session_pair_association_bundle_with_plane_validation, _PATCH_MARKER, True)
    setattr(
        build_session_pair_association_bundle_with_plane_validation,
        "_bayescatrack_original",
        original,
    )
    bridge_impl.build_session_pair_association_bundle = (
        build_session_pair_association_bundle_with_plane_validation
    )


def _function_chain_has_patch(function: Any) -> bool:
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, "_bayescatrack_original", None)
    return False


def _validate_measurement_plane_replacement(
    measurement_session: Any,
    replacement_plane: Any | None,
) -> None:
    if replacement_plane is None:
        return

    measurement_plane = measurement_session.plane_data
    expected_n_rois = int(measurement_plane.n_rois)
    actual_n_rois = int(replacement_plane.n_rois)
    if actual_n_rois != expected_n_rois:
        raise ValueError(
            "measurement_plane_in_reference_frame must preserve the measurement "
            f"session ROI count; expected {expected_n_rois}, got {actual_n_rois}"
        )

    expected_roi_indices = _plane_roi_indices(measurement_plane)
    actual_roi_indices = _plane_roi_indices(replacement_plane)
    if not np.array_equal(np.sort(actual_roi_indices), np.sort(expected_roi_indices)):
        raise ValueError(
            "measurement_plane_in_reference_frame must preserve the measurement "
            "session ROI identities"
        )


def _plane_roi_indices(plane: Any) -> np.ndarray:
    roi_indices = getattr(plane, "roi_indices", None)
    if roi_indices is None:
        return np.arange(int(plane.n_rois), dtype=int)
    return np.asarray(roi_indices, dtype=int)


__all__ = ["install_association_plane_validation"]
