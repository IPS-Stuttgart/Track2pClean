"""Suite2p ``lam`` value validation for weighted-mask loader paths.

Suite2p ``stat.npy`` stores per-pixel ``lam`` weights for each ROI.  The core
loader only uses those values when ``weighted_masks=True``.  In that mode,
non-finite or negative weights should not be returned inside ``roi_masks`` where
downstream overlap and covariance code would sanitize them inconsistently.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_LAM_VALUE_VALIDATION_MARKER = "_bayescatrack_suite2p_lam_value_validation_patch"
_TEXT_OR_BYTES_LIKE_TYPES = (str, bytes, bytearray, memoryview)

_ERROR_MESSAGE = (
    "Suite2p lam values used for weighted masks must be finite and non-negative"
)


def install_suite2p_lam_value_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent validator for weighted Suite2p mask values."""

    original_loader = bridge_impl.load_suite2p_plane
    if getattr(original_loader, _LAM_VALUE_VALIDATION_MARKER, False):
        return

    @wraps(original_loader)
    def _load_suite2p_plane_with_lam_value_validation(
        plane_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if _weighted_masks_requested(args, kwargs):
            _validate_raw_suite2p_lam_values(Path(plane_dir), kwargs)
        plane = original_loader(plane_dir, *args, **kwargs)
        if _weighted_masks_requested(args, kwargs):
            _validate_weighted_mask_values(getattr(plane, "roi_masks", None))
        return plane

    setattr(
        _load_suite2p_plane_with_lam_value_validation,
        _LAM_VALUE_VALIDATION_MARKER,
        True,
    )
    for marker in (
        "_bayescatrack_loader_validation_patch",
        "_bayescatrack_suite2p_coordinate_value_validation_patch",
        "_bayescatrack_suite2p_overlap_value_validation_patch",
        "_bayescatrack_suite2p_iscell_value_validation_patch",
    ):
        if getattr(original_loader, marker, False):
            setattr(_load_suite2p_plane_with_lam_value_validation, marker, True)
    setattr(
        _load_suite2p_plane_with_lam_value_validation,
        "_bayescatrack_original",
        original_loader,
    )
    bridge_impl.load_suite2p_plane = _load_suite2p_plane_with_lam_value_validation


def _weighted_masks_requested(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if args:
        return False
    value = kwargs.get("weighted_masks", False)
    if not isinstance(value, (bool, np.bool_)):
        return False
    return bool(value)


def _validate_raw_suite2p_lam_values(plane_dir: Path, kwargs: dict[str, Any]) -> None:
    stat_path = plane_dir / "stat.npy"
    if not stat_path.exists():
        return

    controls = _parse_selection_controls(kwargs)
    if controls is None:
        return

    stat = np.load(stat_path, allow_pickle=True)
    if stat.ndim != 1:
        return

    keep_mask = _suite2p_keep_mask(
        plane_dir,
        stat,
        include_non_cells=controls["include_non_cells"],
        cell_probability_threshold=controls["cell_probability_threshold"],
    )
    if keep_mask is None:
        return

    for roi_index, roi_stat in enumerate(stat):
        if not keep_mask[roi_index] or "lam" not in roi_stat:
            continue
        raw_lam = np.asarray(roi_stat["lam"])
        selected_lam = _selected_lam_values(
            roi_stat,
            raw_lam,
            exclude_overlapping_pixels=controls["exclude_overlapping_pixels"],
        )
        if selected_lam is not None:
            _validate_lam_values(selected_lam)


def _parse_selection_controls(kwargs: dict[str, Any]) -> dict[str, bool | float] | None:
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


def _selected_lam_values(
    roi_stat: Any,
    raw_lam: np.ndarray,
    *,
    exclude_overlapping_pixels: bool,
) -> np.ndarray | None:
    ypix = np.asarray(roi_stat.get("ypix", ()))
    if raw_lam.shape != ypix.shape:
        return None

    if not exclude_overlapping_pixels or "overlap" not in roi_stat:
        return raw_lam

    overlap = np.asarray(roi_stat["overlap"])
    if overlap.shape != raw_lam.shape:
        return None
    try:
        used = ~np.asarray(overlap, dtype=bool)
    except (TypeError, ValueError):
        return None
    return raw_lam[used]


def _suite2p_keep_mask(
    plane_dir: Path,
    stat: np.ndarray,
    *,
    include_non_cells: bool,
    cell_probability_threshold: float,
) -> np.ndarray | None:
    if include_non_cells:
        return np.ones((int(stat.shape[0]),), dtype=bool)

    iscell_path = plane_dir / "iscell.npy"
    if not iscell_path.exists():
        return np.ones((int(stat.shape[0]),), dtype=bool)

    iscell = np.asarray(np.load(iscell_path, allow_pickle=True))
    if iscell.ndim not in {1, 2} or iscell.shape[0] != stat.shape[0]:
        return None
    if iscell.ndim == 2 and iscell.shape[1] < 1:
        return None

    keep = np.zeros((int(stat.shape[0]),), dtype=bool)
    try:
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
            keep[roi_index] = bool(
                is_cell and probability >= cell_probability_threshold
            )
    except (TypeError, ValueError):
        return None
    return keep


def _validate_weighted_mask_values(roi_masks: Any) -> None:
    _validate_lam_values(roi_masks)


def _validate_lam_values(values: Any) -> None:
    if isinstance(values, _TEXT_OR_BYTES_LIKE_TYPES):
        raise ValueError(_ERROR_MESSAGE)

    try:
        object_values = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc

    if _contains_text_or_bytes_like(object_values):
        raise ValueError(_ERROR_MESSAGE)

    try:
        mask_values = np.asarray(object_values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc

    if not np.all(np.isfinite(mask_values)) or np.any(mask_values < 0.0):
        raise ValueError(_ERROR_MESSAGE)


def _contains_text_or_bytes_like(values: np.ndarray) -> bool:
    return any(
        isinstance(value, _TEXT_OR_BYTES_LIKE_TYPES) for value in values.reshape(-1)
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


__all__ = ["install_suite2p_lam_value_validation"]
