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
        _validate_suite2p_stat_shapes(
            plane_dir,
            include_non_cells=bool(kwargs.get("include_non_cells", False)),
            cell_probability_threshold=float(
                kwargs.get("cell_probability_threshold", 0.5)
            ),
            exclude_overlapping_pixels=bool(
                kwargs.get("exclude_overlapping_pixels", True)
            ),
        )
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
        )


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
) -> None:
    ypix = np.asarray(roi_stat["ypix"])
    xpix = np.asarray(roi_stat["xpix"])
    if ypix.shape != xpix.shape:
        raise ValueError("Suite2p ROI ypix/xpix arrays must have matching shapes")

    if "lam" in roi_stat:
        lam = np.asarray(roi_stat["lam"])
        if lam.shape != ypix.shape:
            raise ValueError("Suite2p ROI lam shape must match ypix/xpix shape")

    if exclude_overlapping_pixels and "overlap" in roi_stat:
        overlap = np.asarray(roi_stat["overlap"])
        if overlap.shape != ypix.shape:
            raise ValueError("Suite2p ROI overlap shape must match ypix/xpix shape")


def _original_load_suite2p_plane() -> Callable[..., Any]:
    original = getattr(
        load_suite2p_plane,
        "_bayescatrack_suite2p_stat_validation_original",
        None,
    )
    if original is None:
        raise RuntimeError("Suite2p stat validation wrapper is not installed")
    return original


__all__ = ["install_suite2p_stat_validation", "load_suite2p_plane"]
