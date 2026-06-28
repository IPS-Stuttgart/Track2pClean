"""Suite2p coordinate and per-pixel-vector validation for subject loaders.

The public ``bayescatrack.load_suite2p_plane`` wrapper already rejects ambiguous
``ypix``/``xpix`` values and malformed per-pixel ``lam``/``overlap`` arrays.
Subject-level loading goes through the lower-level bridge implementation, where
coordinate arrays were converted with ``dtype=int`` before validation and some
per-pixel vectors could be broadcast or ignored.  This patch keeps the
lower-level loader honest before any integer cast or mask reconstruction happens.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_COORDINATE_VALUE_VALIDATION_MARKER = (
    "_bayescatrack_suite2p_coordinate_value_validation_patch"
)


def install_suite2p_coordinate_value_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent pre-cast Suite2p coordinate validator."""

    original_loader = bridge_impl.load_suite2p_plane
    if getattr(original_loader, _COORDINATE_VALUE_VALIDATION_MARKER, False):
        return

    def _load_suite2p_plane_with_coordinate_value_validation(
        plane_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        controls = _parse_selection_controls(kwargs) if not args else None
        if controls is not None:
            _validate_suite2p_coordinate_values(Path(plane_dir), **controls)
        return original_loader(plane_dir, *args, **kwargs)

    setattr(
        _load_suite2p_plane_with_coordinate_value_validation,
        _COORDINATE_VALUE_VALIDATION_MARKER,
        True,
    )
    if getattr(original_loader, "_bayescatrack_loader_validation_patch", False):
        setattr(
            _load_suite2p_plane_with_coordinate_value_validation,
            "_bayescatrack_loader_validation_patch",
            True,
        )
    setattr(
        _load_suite2p_plane_with_coordinate_value_validation,
        "_bayescatrack_original",
        original_loader,
    )
    bridge_impl.load_suite2p_plane = (
        _load_suite2p_plane_with_coordinate_value_validation
    )


def _parse_selection_controls(kwargs: dict[str, Any]) -> dict[str, bool | float] | None:
    """Return validated controls needed to decide which Suite2p ROIs are used.

    Invalid control values are left to the existing loader-control validation so
    this wrapper does not change the established error path for bad flags.
    """

    try:
        return {
            "include_non_cells": _strict_bool(
                kwargs.get("include_non_cells", False), name="include_non_cells"
            ),
            "cell_probability_threshold": _finite_probability(
                kwargs.get("cell_probability_threshold", 0.5),
                name="cell_probability_threshold",
            ),
            "exclude_overlapping_pixels": _strict_bool(
                kwargs.get("exclude_overlapping_pixels", True),
                name="exclude_overlapping_pixels",
            ),
        }
    except ValueError:
        return None


def _validate_suite2p_coordinate_values(
    plane_dir: Path,
    *,
    include_non_cells: bool | float,
    cell_probability_threshold: bool | float,
    exclude_overlapping_pixels: bool | float,
) -> None:
    stat_path = plane_dir / "stat.npy"
    if not stat_path.exists():
        return

    stat = np.load(stat_path, allow_pickle=True)
    if stat.ndim != 1:
        return

    keep_mask = _suite2p_keep_mask(
        plane_dir,
        stat,
        include_non_cells=bool(include_non_cells),
        cell_probability_threshold=float(cell_probability_threshold),
    )
    image_shape = _load_suite2p_ops_image_shape(plane_dir / "ops.npy")
    for roi_index, roi_stat in enumerate(stat):
        if not keep_mask[roi_index]:
            continue

        ypix = np.asarray(roi_stat["ypix"])
        xpix = np.asarray(roi_stat["xpix"])
        overlap = _validate_per_pixel_array_shapes(
            roi_stat,
            roi_index=roi_index,
            ypix=ypix,
            xpix=xpix,
            exclude_overlapping_pixels=bool(exclude_overlapping_pixels),
        )

        if overlap is not None:
            valid = ~np.asarray(overlap, dtype=bool)
            ypix = ypix[valid]
            xpix = xpix[valid]

        _validate_coordinate_array(
            ypix,
            roi_index=roi_index,
            name="ypix",
            axis_size=None if image_shape is None else image_shape[0],
        )
        _validate_coordinate_array(
            xpix,
            roi_index=roi_index,
            name="xpix",
            axis_size=None if image_shape is None else image_shape[1],
        )


def _validate_per_pixel_array_shapes(
    roi_stat: Any,
    *,
    roi_index: int,
    ypix: np.ndarray,
    xpix: np.ndarray,
    exclude_overlapping_pixels: bool,
) -> np.ndarray | None:
    if ypix.shape != xpix.shape:
        raise ValueError(
            f"Suite2p ROI {roi_index} ypix/xpix arrays must have matching shapes"
        )

    if "lam" in roi_stat:
        lam = np.asarray(roi_stat["lam"])
        if lam.shape != ypix.shape:
            raise ValueError(
                f"Suite2p ROI {roi_index} lam shape must match ypix/xpix shape"
            )

    if not exclude_overlapping_pixels or "overlap" not in roi_stat:
        return None

    overlap = np.asarray(roi_stat["overlap"])
    if overlap.shape != ypix.shape:
        raise ValueError(
            f"Suite2p ROI {roi_index} overlap shape must match ypix/xpix shape"
        )
    return overlap


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


def _suite2p_keep_mask(
    plane_dir: Path,
    stat: np.ndarray,
    *,
    include_non_cells: bool,
    cell_probability_threshold: float,
) -> np.ndarray:
    if include_non_cells:
        return np.ones((int(stat.shape[0]),), dtype=bool)

    iscell_path = plane_dir / "iscell.npy"
    if not iscell_path.exists():
        return np.ones((int(stat.shape[0]),), dtype=bool)

    iscell = np.asarray(np.load(iscell_path, allow_pickle=True))
    if iscell.ndim not in {1, 2} or iscell.shape[0] != stat.shape[0]:
        return np.ones((int(stat.shape[0]),), dtype=bool)
    if iscell.ndim == 2 and iscell.shape[1] < 1:
        return np.ones((int(stat.shape[0]),), dtype=bool)

    keep = np.zeros((int(stat.shape[0]),), dtype=bool)
    for roi_index in range(int(stat.shape[0])):
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
        keep[roi_index] = bool(is_cell and probability >= cell_probability_threshold)
    return keep


def _validate_coordinate_array(
    array: np.ndarray,
    *,
    roi_index: int,
    name: str,
    axis_size: int | None,
) -> None:
    if _contains_ambiguous_coordinate_tokens(array):
        _raise_invalid_coordinate_values(roi_index, name)

    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_invalid_coordinate_message(roi_index, name)) from exc

    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        _raise_invalid_coordinate_values(roi_index, name)
    if np.any(numeric < 0.0):
        raise ValueError(_invalid_coordinate_message(roi_index, name))
    if axis_size is not None and np.any(numeric >= float(axis_size)):
        raise ValueError(
            f"Suite2p ROI {roi_index} {name} pixel coordinates must be within image bounds"
        )


def _contains_ambiguous_coordinate_tokens(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.bool_) or array.dtype.kind in {"S", "U"}:
        return True
    if array.dtype != object:
        return False
    return any(isinstance(item, (bool, np.bool_, str, bytes)) for item in array.ravel())


def _raise_invalid_coordinate_values(roi_index: int, name: str) -> None:
    raise ValueError(_invalid_coordinate_message(roi_index, name))


def _invalid_coordinate_message(roi_index: int, name: str) -> str:
    return (
        f"Suite2p ROI {roi_index} {name} must contain finite non-negative "
        "integer pixel coordinates"
    )


def _strict_bool(value: Any, *, name: str) -> bool:
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


__all__ = ["install_suite2p_coordinate_value_validation"]
