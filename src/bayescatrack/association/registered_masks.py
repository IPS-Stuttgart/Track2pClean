"""Utilities for registered ROI masks used by association costs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from bayescatrack.core.bridge import CalciumPlaneData


def empty_registered_roi_mask(plane: CalciumPlaneData) -> np.ndarray:
    """Return a boolean mask selecting registered ROIs with no nonzero pixels."""

    roi_masks = np.asarray(plane.roi_masks)
    if roi_masks.ndim != 3:
        raise ValueError("roi_masks must have shape (n_roi, height, width)")
    return ~np.any(roi_masks != 0, axis=(1, 2))


def drop_empty_registered_masks(
    plane: CalciumPlaneData,
) -> tuple[CalciumPlaneData, np.ndarray]:
    """Drop empty registered ROIs and return the dropped-ROI indicator.

    Image registration can move an ROI completely outside the reference field of
    view.  Those targets are invalid association candidates, but inserting dummy
    pixels for them changes overlap, centroid, and local-evidence components.
    Keep the invalidity explicit instead: compute costs on the non-empty subset
    and expand the resulting matrices back to the original target layout with
    :func:`expand_registered_pairwise_cost_columns`.
    """

    empty_registered_rois = empty_registered_roi_mask(plane)
    if not np.any(empty_registered_rois):
        return plane, empty_registered_rois

    keep_registered_rois = ~empty_registered_rois
    return (
        CalciumPlaneData(
            roi_masks=np.asarray(plane.roi_masks)[keep_registered_rois],
            traces=_slice_optional_roi_array(plane.traces, keep_registered_rois),
            fov=plane.fov,
            spike_traces=_slice_optional_roi_array(
                plane.spike_traces,
                keep_registered_rois,
            ),
            neuropil_traces=_slice_optional_roi_array(
                plane.neuropil_traces,
                keep_registered_rois,
            ),
            cell_probabilities=_slice_optional_roi_array(
                plane.cell_probabilities,
                keep_registered_rois,
            ),
            roi_indices=(
                None
                if plane.roi_indices is None
                else np.asarray(plane.roi_indices, dtype=int)[keep_registered_rois]
            ),
            roi_features={
                key: np.asarray(value)[keep_registered_rois]
                for key, value in plane.roi_features.items()
            },
            source=plane.source,
            plane_name=plane.plane_name,
            ops=plane.ops,
        ),
        empty_registered_rois,
    )


def replace_empty_registered_masks(
    plane: CalciumPlaneData,
) -> tuple[CalciumPlaneData, np.ndarray]:
    """Backward-compatible alias for :func:`drop_empty_registered_masks`.

    The historical implementation inserted one-pixel placeholder masks.  That
    made downstream centroid code accept the plane, but it also contaminated
    overlap and local-evidence components.  The replacement behavior deliberately
    returns a filtered plane plus an explicit invalid-ROI indicator.
    """

    return drop_empty_registered_masks(plane)


def expand_registered_pairwise_cost_columns(
    cost_matrix: np.ndarray,
    empty_registered_rois: np.ndarray,
    *,
    large_cost: float,
) -> np.ndarray:
    """Expand compact pairwise costs and assign ``large_cost`` to empty ROIs."""

    if large_cost <= 0.0:
        raise ValueError("large_cost must be strictly positive")
    return np.asarray(
        expand_registered_roi_columns(
            cost_matrix,
            empty_registered_rois,
            fill_value=float(large_cost),
        ),
        dtype=float,
    )


def expand_registered_pairwise_components(
    components: Mapping[str, np.ndarray],
    empty_registered_rois: np.ndarray,
) -> dict[str, np.ndarray]:
    """Expand two-dimensional component matrices to the original target layout."""

    expanded: dict[str, np.ndarray] = {}
    for key, value in components.items():
        component = np.asarray(value)
        if component.ndim == 2:
            fill_value: float | bool
            if key == "gated":
                fill_value = True
            elif key.endswith("_available") or key.endswith("_valid"):
                fill_value = 0.0
            elif np.issubdtype(component.dtype, np.bool_):
                fill_value = False
            else:
                fill_value = np.nan
            expanded[key] = expand_registered_roi_columns(
                component,
                empty_registered_rois,
                fill_value=fill_value,
            )
        else:
            expanded[key] = component
    return expanded


def expand_registered_roi_columns(
    matrix: np.ndarray,
    empty_registered_rois: np.ndarray,
    *,
    fill_value: float | bool,
) -> np.ndarray:
    """Expand a matrix over non-empty target ROIs back to all registered ROIs."""

    array = np.asarray(matrix)
    empty_registered_rois = np.asarray(empty_registered_rois, dtype=bool)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if empty_registered_rois.ndim != 1:
        raise ValueError("empty_registered_rois must be one-dimensional")

    full_column_count = int(empty_registered_rois.size)
    compact_column_count = int(
        full_column_count - np.count_nonzero(empty_registered_rois)
    )
    if array.shape[1] == full_column_count:
        expanded = np.array(array, copy=True)
        expanded[:, empty_registered_rois] = fill_value
        return expanded
    if array.shape[1] != compact_column_count:
        raise ValueError(
            "matrix must have either one column per original registered ROI or "
            "one column per non-empty registered ROI"
        )

    dtype = array.dtype if np.issubdtype(array.dtype, np.bool_) else float
    expanded = np.full(
        (array.shape[0], full_column_count),
        fill_value,
        dtype=dtype,
    )
    if compact_column_count:
        expanded[:, ~empty_registered_rois] = array
    return expanded


def _slice_optional_roi_array(
    value: np.ndarray | None,
    keep_registered_rois: np.ndarray,
) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)[keep_registered_rois]
