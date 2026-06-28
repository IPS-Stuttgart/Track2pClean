"""Shape validation for aligned-reference Suite2p ROI index vectors."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_aligned_roi_index_validation_patch"


def install_aligned_roi_index_validation(
    reference_module: ModuleType | None = None,
) -> None:
    """Install idempotent validation for ``plane_data.roi_indices`` shape."""

    if reference_module is None:
        from . import (
            reference as reference_module,  # pylint: disable=import-outside-toplevel,reimported
        )

    original_roi_indices_for_plane = (
        reference_module._suite2p_roi_indices_for_plane
    )  # pylint: disable=protected-access
    if getattr(original_roi_indices_for_plane, _PATCH_ATTR, False):
        return

    def _suite2p_roi_indices_for_plane_with_shape_validation(
        plane_data: Any,
    ) -> np.ndarray:
        roi_indices = getattr(plane_data, "roi_indices", None)
        if roi_indices is not None:
            roi_indices_array = np.asarray(roi_indices, dtype=object)
            if roi_indices_array.ndim != 1:
                raise ValueError(
                    "plane_data.roi_indices must be a one-dimensional vector"
                )
        return original_roi_indices_for_plane(plane_data)

    setattr(_suite2p_roi_indices_for_plane_with_shape_validation, _PATCH_ATTR, True)
    setattr(
        _suite2p_roi_indices_for_plane_with_shape_validation,
        "_bayescatrack_original",
        original_roi_indices_for_plane,
    )
    reference_module._suite2p_roi_indices_for_plane = _suite2p_roi_indices_for_plane_with_shape_validation  # pylint: disable=protected-access
