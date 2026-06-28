"""Strict validation for Suite2p ROI stat-array shapes.

The core loader reconstructs ROI masks from Suite2p ``stat.npy`` entries.  Shape
mismatches between ``ypix``/``xpix`` and optional per-pixel arrays should fail
before NumPy indexing can silently ignore malformed metadata or raise a less
diagnostic indexing error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def install_suite2p_stat_validation(bridge_module: Any) -> None:
    """Install an idempotent shape-validation wrapper on ``load_suite2p_plane``."""

    if getattr(bridge_module, "_bayescatrack_suite2p_stat_validation_patch", False):
        return

    original_load_suite2p_plane = bridge_module.load_suite2p_plane
    setattr(
        load_suite2p_plane,
        "_bayescatrack_suite2p_stat_validation_original",
        original_load_suite2p_plane,
    )
    bridge_module.load_suite2p_plane = load_suite2p_plane
    setattr(bridge_module, "_bayescatrack_suite2p_stat_validation_patch", True)


def load_suite2p_plane(plane_dir: str | Path, *args: Any, **kwargs: Any) -> Any:
    """Validate Suite2p per-pixel stat-array shapes before loading a plane."""

    original = _original_load_suite2p_plane()
    if not args:
        include_non_cells = _strict_bool(
            kwargs.get("include_non_cells", False), name="include_non_cells"
        )
        cell_probability_threshold = _finite_probability(
            kwargs.get("cell_probability_threshold", 0.5),
            name="cell_probability_threshold",
        )
        exclude_overlapping_pixels = _strict_bool(
            kwargs.get("exclude_overlapping_pixels", True),
            name="exclude_overlapping_pixels",
        )
        weighted_masks = _strict_bool(
            kwargs.get("weighted_masks", False), name="weighted_masks"
        )
        load_traces = _strict_python_bool(
            kwargs.get("load_traces", True), name="load_traces"
        )
        load_spike_traces = _strict_python_bool(
            kwargs.get("load_spike_traces", True), name="load_spike_traces"
        )
        load_neuropil_traces = _strict_python_bool(
            kwargs.get("load_neuropil_traces", False), name="load_neuropil_traces"
        )
        _validate_suite2p_stat_shapes(
            plane_dir,
            include_non_cells=include_non_cells,
            cell_probability_threshold=cell_probability_threshold,
            exclude_overlapping_pixels=exclude_overlapping_pixels,
        )
        kwargs = {
            **kwargs,
            "include_non_cells": include_non_cells,
            "cell_probability_threshold": cell_probability_threshold,
            "exclude_overlapping_pixels": exclude_overlapping_pixels,
            "weighted_masks": weighted_masks,
            "load_traces": load_traces,
            "load_spike_traces": load_spike_traces,
            "load_neuropil_traces": load_neuropil_traces,
        }
    return original(plane_dir, *args, **kwargs)


def _validate_suite2p_stat_shapes(
    plane_dir: str | Path,
    *,
    include_non_cells: bool,
    cell_probability_threshold: float,
    exclude_overlapping_pixels: bool,
) -> None:
    plane_path = Path(plane_dir)
    stat = np.load(plane_path / "stat.npy", allow_pickle=True)
    if stat.ndim != 1:
        return

    iscell = _load_iscell_if_shape_compatible(plane_path / "iscell.npy", stat.shape[0])
    image_shape = _load_suite2p_ops_image_shape(plane_path / "ops.npy")
    for roi_index, roi_stat in enumerate(stat):
        if not _suite2p_roi_is_selected(
            roi_index,
            iscell,
            include_non_cells=include_non_cells,
            cell_probability_threshold=cell_probability_threshold,
        ):
            continue
        _validate_one_suite2p_roi_stat(
            roi_stat,
            exclude_overlapping_pixels=exclude_overlapping_pixels,
            image_shape=image_shape,
        )


def _load_suite2p_ops_image_shape(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    ops = np.load(path, allow_pickle=True).item()
    try:
        raw_ly = ops["Ly"]
        raw_lx = ops["Lx"]
    except (KeyError, TypeError):
        return None
    return (
        _validate_positive_image_dimension(raw_ly, name="Ly"),
        _validate_positive_image_dimension(raw_lx, name="Lx"),
    )


def _validate_positive_image_dimension(value: Any, *, name: str) -> int:
    message = f"Suite2p ops {name} must be a positive integer image dimension"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric != np.floor(numeric):
        raise ValueError(message)
    return int(numeric)


def _load_iscell_if_shape_compatible(path: Path, n_rois: int) -> np.ndarray | None:
    if not path.exists():
        return None
    iscell = np.asarray(np.load(path, allow_pickle=True))
    if iscell.ndim not in {1, 2} or iscell.shape[0] != int(n_rois):
        return None
    if iscell.ndim == 2 and iscell.shape[1] < 1:
        return None
    return iscell


def _suite2p_roi_is_selected(
    roi_index: int,
    iscell: np.ndarray | None,
    *,
    include_non_cells: bool,
    cell_probability_threshold: float,
) -> bool:
    if iscell is None or include_non_cells:
        return True

    if iscell.ndim == 2:
        is_cell = bool(iscell[roi_index, 0])
        probability = (
            float(iscell[roi_index, 1])
            if iscell.shape[1] > 1
            else float(iscell[roi_index, 0])
        )
    else:
        is_cell = bool(iscell[roi_index])
        probability = float(iscell[roi_index])

    return is_cell and probability >= cell_probability_threshold


def _validate_one_suite2p_roi_stat(
    roi_stat: Any,
    *,
    exclude_overlapping_pixels: bool,
    image_shape: tuple[int, int] | None,
) -> None:
    ypix = _validate_integer_pixel_coordinate_array(roi_stat["ypix"], name="ypix")
    xpix = _validate_integer_pixel_coordinate_array(roi_stat["xpix"], name="xpix")
    if ypix.shape != xpix.shape:
        raise ValueError("Suite2p ROI ypix/xpix arrays must have matching shapes")

    _validate_pixel_coordinate_bounds(
        ypix,
        name="ypix",
        axis_size=None if image_shape is None else image_shape[0],
    )
    _validate_pixel_coordinate_bounds(
        xpix,
        name="xpix",
        axis_size=None if image_shape is None else image_shape[1],
    )

    if "lam" in roi_stat:
        lam = np.asarray(roi_stat["lam"])
        if lam.shape != ypix.shape:
            raise ValueError("Suite2p ROI lam shape must match ypix/xpix shape")

    if exclude_overlapping_pixels and "overlap" in roi_stat:
        overlap = np.asarray(roi_stat["overlap"])
        if overlap.shape != ypix.shape:
            raise ValueError("Suite2p ROI overlap shape must match ypix/xpix shape")


def _validate_pixel_coordinate_bounds(
    coordinates: np.ndarray,
    *,
    name: str,
    axis_size: int | None,
) -> None:
    if coordinates.size == 0:
        return
    if np.any(coordinates < 0):
        raise ValueError(f"Suite2p ROI {name} pixel coordinates must be non-negative")
    if axis_size is not None and np.any(coordinates >= axis_size):
        raise ValueError(
            f"Suite2p ROI {name} pixel coordinates must be within image bounds"
        )


def _validate_integer_pixel_coordinate_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(
            f"Suite2p ROI {name} must be a one-dimensional pixel-coordinate array"
        )
    if _contains_ambiguous_pixel_coordinate_tokens(array):
        raise ValueError(
            f"Suite2p ROI {name} must contain finite integer pixel coordinates"
        )
    if np.issubdtype(array.dtype, np.integer):
        return array

    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Suite2p ROI {name} must contain finite integer pixel coordinates"
        ) from exc

    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(
            f"Suite2p ROI {name} must contain finite integer pixel coordinates"
        )
    return numeric.astype(int, copy=False)


def _contains_ambiguous_pixel_coordinate_tokens(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.bool_) or array.dtype.kind in {"S", "U"}:
        return True
    if array.dtype != object:
        return False
    return any(isinstance(item, (bool, np.bool_, str, bytes)) for item in array.ravel())


def _original_load_suite2p_plane() -> Callable[..., Any]:
    original = getattr(
        load_suite2p_plane,
        "_bayescatrack_suite2p_stat_validation_original",
        None,
    )
    if original is None:
        raise RuntimeError("Suite2p stat validation wrapper is not installed")
    return original


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_python_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_probability(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite probability")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return numeric


__all__ = ["install_suite2p_stat_validation", "load_suite2p_plane"]
