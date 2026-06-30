"""Strict validation for FOV subpixel translation controls.

The wrappers normalize public subpixel shift vectors before low-level resampling
so malformed, boolean, non-finite, and string-like values fail with ``ValueError``.
They also validate mask-interpolation controls before delegation.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_subpixel_shift_validation_patch"
_MASK_INTERPOLATION_PATCH_MARKER = (
    "_bayescatrack_fov_mask_interpolation_validation_patch"
)
_FLOAT_CONTROL_PATCH_MARKER = "_bayescatrack_fov_float_control_validation_patch"
_INTERPOLATION_ORDER_PATCH_MARKER = (
    "_bayescatrack_fov_interpolation_order_validation_patch"
)
_SHIFT_ERROR = "shift_yx must contain exactly two finite numeric values"
_MASK_INTERPOLATION_ERROR = "mask_interpolation must be either 'nearest' or 'bilinear'"
_FLOAT_CONTROL_ERROR = "{name} must be a finite non-negative value"
_INTERPOLATION_ORDER_ERROR = (
    "subpixel interpolation order must be an integer between 0 and 5"
)


def install_fov_subpixel_shift_validation() -> None:
    """Install idempotent validation around FOV subpixel translation controls."""

    from . import (
        fov_registration as _fov_registration,  # pylint: disable=import-outside-toplevel
    )

    _wrap_shift_argument(_fov_registration, "apply_subpixel_image_translation")
    _wrap_shift_argument(_fov_registration, "apply_subpixel_roi_mask_translation")
    _wrap_mask_interpolation_validation(_fov_registration)
    _wrap_subpixel_interpolation_order_validation(_fov_registration)
    _wrap_finite_nonnegative_float_validation(_fov_registration)


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


def _wrap_mask_interpolation_validation(module: Any) -> None:
    original = module._validate_mask_interpolation  # pylint: disable=protected-access
    if getattr(original, _MASK_INTERPOLATION_PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(interpolation: Any) -> None:
        if not isinstance(interpolation, str):
            raise ValueError(_MASK_INTERPOLATION_ERROR)
        return original(interpolation)

    setattr(wrapper, _MASK_INTERPOLATION_PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    module._validate_mask_interpolation = wrapper  # pylint: disable=protected-access


def _wrap_subpixel_interpolation_order_validation(module: Any) -> None:
    original = module._validate_subpixel_interpolation_order  # pylint: disable=protected-access
    if getattr(original, _INTERPOLATION_ORDER_PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(interpolation_order: Any) -> int:
        try:
            return original(interpolation_order)
        except (ValueError, OverflowError) as exc:
            raise ValueError(_INTERPOLATION_ORDER_ERROR) from exc

    setattr(wrapper, _INTERPOLATION_ORDER_PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    module._validate_subpixel_interpolation_order = wrapper  # pylint: disable=protected-access


def _wrap_finite_nonnegative_float_validation(module: Any) -> None:
    original = module._finite_nonnegative_float  # pylint: disable=protected-access
    if getattr(original, _FLOAT_CONTROL_PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(value: Any, *, name: str) -> float:
        try:
            return original(value, name=name)
        except OverflowError as exc:
            raise ValueError(_FLOAT_CONTROL_ERROR.format(name=name)) from exc

    setattr(wrapper, _FLOAT_CONTROL_PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    module._finite_nonnegative_float = wrapper  # pylint: disable=protected-access


def _normalize_subpixel_shift_yx(shift_yx: Any) -> np.ndarray:
    if isinstance(shift_yx, (str, bytes, bytearray)):
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
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(_SHIFT_ERROR)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(_SHIFT_ERROR)
    return numeric_value


__all__ = ["install_fov_subpixel_shift_validation"]
