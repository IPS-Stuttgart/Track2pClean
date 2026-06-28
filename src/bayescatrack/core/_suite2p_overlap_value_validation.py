"""Suite2p overlap-vector value validation for loader paths.

Suite2p ``stat.npy`` stores an optional per-pixel ``overlap`` vector that should be
boolean.  The core loader uses it to drop overlapped pixels before mask
reconstruction.  Coercing arbitrary object/string/numeric arrays through
``dtype=bool`` is unsafe because values such as ``"False"`` become ``True`` and can
silently remove valid ROI pixels.  This patch rejects non-boolean overlap vectors
before any bool coercion is used by the loader stack.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_OVERLAP_VALUE_VALIDATION_MARKER = (
    "_bayescatrack_suite2p_overlap_value_validation_patch"
)


def install_suite2p_overlap_value_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent pre-cast Suite2p overlap-value validator."""

    original_loader = bridge_impl.load_suite2p_plane
    if getattr(original_loader, _OVERLAP_VALUE_VALIDATION_MARKER, False):
        return

    @wraps(original_loader)
    def _load_suite2p_plane_with_overlap_value_validation(
        plane_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        controls = _parse_selection_controls(kwargs) if not args else None
        if controls is not None and controls["exclude_overlapping_pixels"]:
            _validate_suite2p_overlap_values(
                Path(plane_dir),
                include_non_cells=controls["include_non_cells"],
                cell_probability_threshold=controls["cell_probability_threshold"],
            )
        return original_loader(plane_dir, *args, **kwargs)

    setattr(
        _load_suite2p_plane_with_overlap_value_validation,
        _OVERLAP_VALUE_VALIDATION_MARKER,
        True,
    )
    for marker in (
        "_bayescatrack_loader_validation_patch",
        "_bayescatrack_suite2p_coordinate_value_validation_patch",
    ):
        if getattr(original_loader, marker, False):
            setattr(_load_suite2p_plane_with_overlap_value_validation, marker, True)
    setattr(
        _load_suite2p_plane_with_overlap_value_validation,
        "_bayescatrack_original",
        original_loader,
    )
    bridge_impl.load_suite2p_plane = _load_suite2p_plane_with_overlap_value_validation


def _parse_selection_controls(kwargs: dict[str, Any]) -> dict[str, bool | float] | None:
    """Return validated controls needed to decide which Suite2p ROIs are used."""

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


def _validate_suite2p_overlap_values(
    plane_dir: Path,
    *,
    include_non_cells: bool | float,
    cell_probability_threshold: bool | float,
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
    for roi_index, roi_stat in enumerate(stat):
        if not keep_mask[roi_index] or "overlap" not in roi_stat:
            continue
        overlap = np.asarray(roi_stat["overlap"])
        if not _is_boolean_overlap_array(overlap):
            raise ValueError(
                f"Suite2p ROI {roi_index} overlap must contain boolean values; "
                f"got dtype {overlap.dtype}"
            )


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


def _is_boolean_overlap_array(overlap: np.ndarray) -> bool:
    if np.issubdtype(overlap.dtype, np.bool_):
        return True
    if overlap.dtype != object:
        return False
    return all(isinstance(value, (bool, np.bool_)) for value in overlap.ravel())


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


__all__ = ["install_suite2p_overlap_value_validation"]
