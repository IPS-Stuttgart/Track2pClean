"""Strict validation for FOV-translation output shapes.

The low-level FOV translation helpers allocate destination images from
``output_shape`` after direct tuple expansion or ``int(...)`` coercion.  That
means malformed dimensions such as ``True`` or ``4.5`` can be silently
reinterpreted before the image/ROI warp is applied.  This package-level hook
normalizes optional output shapes once, preserving integer-like dimensions while
rejecting booleans, fractional values, non-finite values, strings, zeros, and
malformed shapes.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_translation_output_shape_validation_patch"
_OUTPUT_SHAPE_ERROR = "output_shape must contain exactly two positive integer dimensions"


def install_fov_translation_output_shape_validation() -> None:
    """Install idempotent validation around FOV translation output shapes."""

    from . import fov_registration as _fov_registration  # pylint: disable=import-outside-toplevel

    _wrap_output_shape_kwarg(
        _fov_registration,
        "apply_integer_image_translation",
    )
    _wrap_output_shape_kwarg(
        _fov_registration,
        "apply_subpixel_image_translation",
    )
    _wrap_output_shape_kwarg(
        _fov_registration,
        "apply_integer_roi_mask_translation",
    )
    _wrap_output_shape_kwarg(
        _fov_registration,
        "apply_subpixel_roi_mask_translation",
    )


def _wrap_output_shape_kwarg(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        normalized_kwargs = _normalize_output_shape_kwarg(kwargs)
        return original(*args, **normalized_kwargs)

    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    setattr(module, function_name, wrapper)


def _normalize_output_shape_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "output_shape" not in kwargs or kwargs["output_shape"] is None:
        return dict(kwargs)
    normalized_kwargs = dict(kwargs)
    normalized_kwargs["output_shape"] = _normalize_output_shape(
        normalized_kwargs["output_shape"]
    )
    return normalized_kwargs


def _normalize_output_shape(output_shape: Any) -> tuple[int, int]:
    if isinstance(output_shape, (str, bytes)):
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    try:
        shape_array = np.asarray(output_shape, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_OUTPUT_SHAPE_ERROR) from exc
    if shape_array.shape != (2,):
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    height, width = (
        _normalize_output_shape_dimension(value) for value in shape_array.tolist()
    )
    return height, width


def _normalize_output_shape_dimension(value: Any) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(_OUTPUT_SHAPE_ERROR)

    try:
        dimension = int(operator.index(value))
    except TypeError:
        if isinstance(value, (float, np.floating)):
            numeric_value = float(value)
            if np.isfinite(numeric_value) and numeric_value.is_integer():
                dimension = int(numeric_value)
            else:
                raise ValueError(_OUTPUT_SHAPE_ERROR) from None
        else:
            raise ValueError(_OUTPUT_SHAPE_ERROR) from None

    if dimension <= 0:
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    return dimension


__all__ = ["install_fov_translation_output_shape_validation"]
