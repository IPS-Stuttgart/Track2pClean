"""Suite2p ``iscell.npy`` value validation for loader paths.

Suite2p stores cell-classifier flags/probabilities as numeric values.  The
loader stack uses these values to decide which ROIs should be reconstructed, so
invalid values must be rejected before Python truthiness or float clipping can
silently change ROI selection.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_ISCELL_VALUE_VALIDATION_MARKER = (
    "_bayescatrack_suite2p_iscell_value_validation_patch"
)


def install_suite2p_iscell_value_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent pre-cast Suite2p ``iscell.npy`` value validator."""

    original_loader = bridge_impl.load_suite2p_plane
    if getattr(original_loader, _ISCELL_VALUE_VALIDATION_MARKER, False):
        return

    @wraps(original_loader)
    def _load_suite2p_plane_with_iscell_value_validation(
        plane_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        _validate_suite2p_iscell_values(Path(plane_dir))
        return original_loader(plane_dir, *args, **kwargs)

    setattr(
        _load_suite2p_plane_with_iscell_value_validation,
        _ISCELL_VALUE_VALIDATION_MARKER,
        True,
    )
    for marker in (
        "_bayescatrack_loader_validation_patch",
        "_bayescatrack_suite2p_coordinate_value_validation_patch",
        "_bayescatrack_suite2p_overlap_value_validation_patch",
    ):
        if getattr(original_loader, marker, False):
            setattr(_load_suite2p_plane_with_iscell_value_validation, marker, True)
    setattr(
        _load_suite2p_plane_with_iscell_value_validation,
        "_bayescatrack_original",
        original_loader,
    )
    bridge_impl.load_suite2p_plane = _load_suite2p_plane_with_iscell_value_validation


def _validate_suite2p_iscell_values(plane_dir: Path) -> None:
    iscell_path = plane_dir / "iscell.npy"
    if not iscell_path.exists():
        return

    iscell = np.asarray(np.load(iscell_path, allow_pickle=True))
    if iscell.ndim not in {1, 2}:
        return
    if iscell.ndim == 2 and iscell.shape[1] < 1:
        return

    if iscell.ndim == 1:
        _validate_probability_like_values(iscell, column_name="values")
        return

    _validate_probability_like_values(iscell[:, 0], column_name="cell-flag column")
    if iscell.shape[1] > 1:
        _validate_probability_like_values(
            iscell[:, 1],
            column_name="cell-probability column",
        )


def _validate_probability_like_values(values: np.ndarray, *, column_name: str) -> None:
    raw_values = np.asarray(values)
    if _contains_text_tokens(raw_values):
        raise ValueError(_invalid_iscell_message(column_name))

    try:
        numeric_values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_invalid_iscell_message(column_name)) from exc

    if not np.all(np.isfinite(numeric_values)) or np.any(
        (numeric_values < 0.0) | (numeric_values > 1.0)
    ):
        raise ValueError(_invalid_iscell_message(column_name))


def _contains_text_tokens(values: np.ndarray) -> bool:
    if values.dtype.kind in {"S", "U"}:
        return True
    if values.dtype != object:
        return False
    return any(isinstance(value, (str, bytes)) for value in values.ravel())


def _invalid_iscell_message(column_name: str) -> str:
    return f"iscell.npy {column_name} must contain finite numbers in [0, 1]"


__all__ = ["install_suite2p_iscell_value_validation"]
