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


_ERROR_MESSAGE = (
    "Suite2p lam values used for weighted masks must be finite and non-negative"
)


def install_suite2p_lam_value_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent post-load validator for weighted Suite2p masks."""

    original_loader = bridge_impl.load_suite2p_plane
    if getattr(original_loader, _LAM_VALUE_VALIDATION_MARKER, False):
        return

    @wraps(original_loader)
    def _load_suite2p_plane_with_lam_value_validation(
        plane_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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


def _validate_weighted_mask_values(roi_masks: Any) -> None:
    try:
        mask_values = np.asarray(roi_masks, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc

    if not np.all(np.isfinite(mask_values)) or np.any(mask_values < 0.0):
        raise ValueError(_ERROR_MESSAGE)


__all__ = ["install_suite2p_lam_value_validation"]
