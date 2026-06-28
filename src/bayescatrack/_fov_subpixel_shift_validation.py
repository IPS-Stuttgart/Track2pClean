"""Strict validation for FOV subpixel translation shifts.

The low-level subpixel translation helpers previously coerced ``shift_yx`` with
``np.asarray(..., dtype=float).reshape(2)``.  Boolean values therefore became
numeric one-pixel or zero-pixel shifts before image or ROI-mask resampling, and
malformed shapes surfaced as low-level reshape errors.  This package-level hook
keeps ordinary numeric shifts working while rejecting booleans, non-finite
values, and malformed shift vectors at the API boundary.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_subpixel_shift_validation_patch"
_SHIFT_ERROR = "shift_yx must contain exactly two finite numeric values"


def install_fov_subpixel_shift_validation() -> None:
    """Install idempotent validation around FOV subpixel translation shifts."""

    from . import fov_registration as _fov_registration  # pylint: disable=import-outside-toplevel

    _wrap_shift_argument(_fov_registration, "apply_subpixel_image_translation")
    _wrap_shift_argument(_fov_registration, "apply_subpixel_roi_mask_translation")


def _wrap_shift_argument(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(source: Any, shift_yx: Any, *args: Any, **kwargs: Any) -> Any:
        return original(source, _normalize_subpixel_shift_yx(shift_yx), *args, **kwargs)

    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    setattr(module, function_name, wrapper)


def _normalize_subpixel_shift_yx(shift_yx: Any) -> np.ndarray:
    if isinstance(shift_yx, (str, bytes)):
        raise ValueError(_SHIFT_ERROR)
    try:
        shift_array = np.asarray(shift_yx, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc

    if shift_array.shape != (2,):
        raise ValueError(_SHIFT_ERROR)

    return np.asarray(
        [_normalize_subpixel_shift_component(value) for value in shift_array.tolist()],
        dtype=float,
    )


def _normalize_subpixel_shift_component(value: Any) -> float:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(_SHIFT_ERROR)
        value = value.item()
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(_SHIFT_ERROR)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(_SHIFT_ERROR)
    return numeric_value


__all__ = ["install_fov_subpixel_shift_validation"]
