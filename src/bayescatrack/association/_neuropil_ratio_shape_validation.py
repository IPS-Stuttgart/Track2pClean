"""Validation patch for neuropil-ratio activity feature shapes."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import activity_similarity as _activity_similarity

_PATCH_MARKER = "_bayescatrack_neuropil_ratio_shape_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_neuropil_ratio_shape_validation_original"


def install_neuropil_ratio_shape_validation() -> None:
    """Prevent mismatched neuropil traces from being broadcast across ROIs."""

    if getattr(_activity_similarity, _PATCH_MARKER, False):
        return

    original = (
        _activity_similarity._per_roi_neuropil_ratio
    )  # pylint: disable=protected-access

    def validated_per_roi_neuropil_ratio(
        fluorescence_traces: np.ndarray | None,
        neuropil_traces: np.ndarray | None,
        *,
        similarity_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if _has_mismatched_roi_axis(fluorescence_traces, neuropil_traces):
            return None
        return original(
            fluorescence_traces,
            neuropil_traces,
            similarity_epsilon=similarity_epsilon,
        )

    validated_per_roi_neuropil_ratio.__name__ = original.__name__
    validated_per_roi_neuropil_ratio.__qualname__ = original.__qualname__
    setattr(validated_per_roi_neuropil_ratio, _ORIGINAL_ATTR, original)
    _activity_similarity._per_roi_neuropil_ratio = (
        validated_per_roi_neuropil_ratio  # pylint: disable=protected-access
    )
    setattr(_activity_similarity, _PATCH_MARKER, True)


def _has_mismatched_roi_axis(
    fluorescence_traces: Any,
    neuropil_traces: Any,
) -> bool:
    if fluorescence_traces is None or neuropil_traces is None:
        return False

    fluorescence_array = np.asarray(fluorescence_traces)
    neuropil_array = np.asarray(neuropil_traces)
    if fluorescence_array.ndim != 2 or neuropil_array.ndim != 2:
        return False
    return int(fluorescence_array.shape[0]) != int(neuropil_array.shape[0])


__all__ = ["install_neuropil_ratio_shape_validation"]
