"""Runtime validation patches for CalciumPlaneData ROI index metadata."""

from __future__ import annotations

from typing import Any

import numpy as np


def install_calcium_plane_roi_index_validation(
    calcium_plane_data_cls: type[Any],
) -> None:
    """Install an idempotent ROI-index validator on CalciumPlaneData."""

    original_post_init = calcium_plane_data_cls.__post_init__
    if getattr(original_post_init, "_bayescatrack_roi_index_validation_patch", False):
        return

    def _post_init_with_roi_index_validation(self: Any) -> None:
        _validate_roi_indices(
            getattr(self, "roi_indices", None),
            getattr(self, "roi_masks", None),
        )
        original_post_init(self)

    setattr(
        _post_init_with_roi_index_validation,
        "_bayescatrack_roi_index_validation_patch",
        True,
    )
    setattr(
        _post_init_with_roi_index_validation,
        "_bayescatrack_original",
        original_post_init,
    )
    calcium_plane_data_cls.__post_init__ = _post_init_with_roi_index_validation


def _validate_roi_indices(roi_indices: Any, roi_masks: Any) -> None:
    if roi_indices is None:
        return

    raw_indices = np.asarray(roi_indices, dtype=object)
    mask_array = np.asarray(roi_masks)
    if mask_array.ndim >= 1 and raw_indices.shape != (int(mask_array.shape[0]),):
        raise ValueError("roi_indices must have shape (n_roi,)")

    for index, value in np.ndenumerate(raw_indices):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"roi_indices must contain integer ROI indices, got boolean at {index}"
            )
        if isinstance(value, (str, np.str_)):
            raise ValueError(
                f"roi_indices must contain integer ROI indices, got text at {index}"
            )
        if isinstance(value, (float, np.floating)) and (
            not np.isfinite(value) or not float(value).is_integer()
        ):
            raise ValueError("roi_indices must contain integer ROI indices")

    try:
        normalized = np.asarray(raw_indices, dtype=int)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("roi_indices must contain integer ROI indices") from exc

    if np.any(normalized < 0):
        raise ValueError("roi_indices must contain non-negative indices")
    if len(set(normalized.reshape(-1).tolist())) != int(normalized.size):
        raise ValueError("roi_indices must contain unique indices")
