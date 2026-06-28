"""Keep replacement-mask FOV metadata shape-valid."""

from __future__ import annotations

from typing import Any

import numpy as np


def install_with_replaced_masks_fov_validation(
    calcium_plane_data_cls: type[Any],
) -> None:
    """Patch ``with_replaced_masks`` so stale FOV images are not reused."""

    original_method = calcium_plane_data_cls.with_replaced_masks
    if getattr(original_method, "_track2pclean_fov_shape_safe", False):
        return

    def with_replaced_masks(
        self: Any,
        roi_masks: np.ndarray,
        *,
        fov: np.ndarray | None = None,
        source: str | None = None,
        plane_name: str | None = None,
        ops: dict[str, Any] | None = None,
    ) -> Any:
        roi_masks_array = np.asarray(roi_masks)
        if roi_masks_array.ndim != 3:
            raise ValueError("roi_masks must have shape (n_roi, height, width)")
        if roi_masks_array.shape[0] != self.n_rois:
            raise ValueError(
                "roi_masks must preserve the number of ROIs when replacing masks"
            )

        replacement_fov = fov
        if (
            replacement_fov is None
            and self.fov is not None
            and tuple(np.asarray(self.fov).shape) == tuple(roi_masks_array.shape[1:])
        ):
            replacement_fov = self.fov

        return calcium_plane_data_cls(
            roi_masks=roi_masks_array,
            traces=self.traces,
            fov=replacement_fov,
            spike_traces=self.spike_traces,
            neuropil_traces=self.neuropil_traces,
            cell_probabilities=self.cell_probabilities,
            roi_indices=self.roi_indices,
            roi_features=self.roi_features,
            source=self.source if source is None else source,
            plane_name=self.plane_name if plane_name is None else plane_name,
            ops=self.ops if ops is None else ops,
        )

    with_replaced_masks._track2pclean_fov_shape_safe = True  # type: ignore[attr-defined]
    calcium_plane_data_cls.with_replaced_masks = with_replaced_masks
