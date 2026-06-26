"""Strict Suite2p ``iscell.npy`` value validation."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_suite2p_iscell_value_validation_patch"


def install_suite2p_iscell_value_validation(bridge_module: Any) -> None:
    """Install an idempotent pre-load validator for Suite2p ``iscell.npy`` values."""

    original = bridge_module.load_suite2p_plane
    if getattr(original, _PATCH_ATTR, False):
        return

    @wraps(original)
    def load_suite2p_plane(plane_dir: str | Path, *args: Any, **kwargs: Any) -> Any:
        _validate_suite2p_iscell_values(Path(plane_dir))
        return original(plane_dir, *args, **kwargs)

    setattr(load_suite2p_plane, _PATCH_ATTR, True)
    setattr(load_suite2p_plane, "_bayescatrack_original", original)
    bridge_module.load_suite2p_plane = load_suite2p_plane


def _validate_suite2p_iscell_values(plane_dir: Path) -> None:
    iscell_path = plane_dir / "iscell.npy"
    if not iscell_path.exists():
        return

    n_rois = _load_stat_roi_count(plane_dir / "stat.npy")
    iscell = np.asarray(np.load(iscell_path, allow_pickle=True))
    if iscell.ndim not in {1, 2}:
        return
    if n_rois is not None and iscell.shape[0] != n_rois:
        return
    if iscell.ndim == 2 and iscell.shape[1] < 1:
        return

    if iscell.ndim == 1:
        _validate_binary_values(
            iscell,
            message="Suite2p iscell values must contain finite binary values (0 or 1)",
        )
        return

    _validate_binary_values(
        iscell[:, 0],
        message="Suite2p iscell cell flags must contain finite binary values (0 or 1)",
    )
    if iscell.shape[1] > 1:
        _validate_probability_values(
            iscell[:, 1],
            message="Suite2p iscell probabilities must contain finite probabilities between 0 and 1",
        )


def _load_stat_roi_count(path: Path) -> int | None:
    if not path.exists():
        return None
    stat = np.load(path, allow_pickle=True)
    if stat.ndim != 1:
        return None
    return int(stat.shape[0])


def _validate_binary_values(values: np.ndarray, *, message: str) -> None:
    for value in np.asarray(values, dtype=object).reshape(-1):
        numeric = _finite_numeric_value(value, message=message)
        if numeric not in {0.0, 1.0}:
            raise ValueError(message)


def _validate_probability_values(values: np.ndarray, *, message: str) -> None:
    for value in np.asarray(values, dtype=object).reshape(-1):
        numeric = _finite_numeric_value(value, message=message)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(message)


def _finite_numeric_value(value: Any, *, message: str) -> float:
    if isinstance(value, (str, bytes)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


__all__ = ["install_suite2p_iscell_value_validation"]
