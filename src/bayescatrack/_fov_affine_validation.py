"""Strict validation for FOV-affine warp controls.

The affine FOV fallback previously passed user-supplied affine matrices and
output shapes directly into NumPy coercions.  Non-finite transform entries can
therefore collapse the warp into an all-fill output, and malformed shapes such
as booleans or fractional values can be silently reinterpreted through
``int(...)``.  This package-level hook preserves normal integer-like shapes but
fails fast for malformed geometry before image or ROI evidence is warped.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_IMAGE_PATCH_MARKER = "_bayescatrack_fov_affine_image_warp_validation_patch"
_MASK_PATCH_MARKER = "_bayescatrack_fov_affine_roi_mask_warp_validation_patch"
_MATRIX_ERROR = "matrix_xy must be a finite 2-by-3 affine matrix"
_OUTPUT_SHAPE_ERROR = (
    "output_shape must contain exactly two non-negative integer values"
)


def install_fov_affine_warp_validation() -> None:
    """Install idempotent validation around FOV-affine image and mask warps."""

    from . import (
        fov_affine_registration as _fov_affine_registration,  # pylint: disable=import-outside-toplevel
    )

    original_image_warp = _fov_affine_registration.apply_affine_image_warp
    if not getattr(original_image_warp, _IMAGE_PATCH_MARKER, False):

        @wraps(original_image_warp)
        def apply_affine_image_warp_with_validation(
            image: Any,
            matrix_xy: Any,
            *args: Any,
            **kwargs: Any,
        ) -> np.ndarray:
            return original_image_warp(
                image,
                _normalize_affine_matrix_xy(matrix_xy),
                *args,
                **_normalize_output_shape_kwarg(kwargs),
            )

        _mark_patch(
            apply_affine_image_warp_with_validation,
            original_image_warp,
            _IMAGE_PATCH_MARKER,
        )
        _fov_affine_registration.apply_affine_image_warp = (
            apply_affine_image_warp_with_validation
        )

    original_mask_warp = _fov_affine_registration.apply_affine_roi_mask_warp
    if not getattr(original_mask_warp, _MASK_PATCH_MARKER, False):

        @wraps(original_mask_warp)
        def apply_affine_roi_mask_warp_with_validation(
            roi_masks: Any,
            matrix_xy: Any,
            *args: Any,
            **kwargs: Any,
        ) -> np.ndarray:
            return original_mask_warp(
                roi_masks,
                _normalize_affine_matrix_xy(matrix_xy),
                *args,
                **_normalize_output_shape_kwarg(kwargs),
            )

        _mark_patch(
            apply_affine_roi_mask_warp_with_validation,
            original_mask_warp,
            _MASK_PATCH_MARKER,
        )
        _fov_affine_registration.apply_affine_roi_mask_warp = (
            apply_affine_roi_mask_warp_with_validation
        )


def _mark_patch(wrapper: Any, original: Any, marker: str) -> None:
    setattr(wrapper, marker, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_output_shape_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "output_shape" not in kwargs:
        return kwargs
    normalized_kwargs = dict(kwargs)
    normalized_kwargs["output_shape"] = _normalize_output_shape(kwargs["output_shape"])
    return normalized_kwargs


def _normalize_output_shape(output_shape: Any) -> tuple[int, int]:
    try:
        shape_array = np.asarray(output_shape, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_OUTPUT_SHAPE_ERROR) from exc

    flat_shape = shape_array.reshape(-1)
    if flat_shape.size != 2:
        raise ValueError(_OUTPUT_SHAPE_ERROR)

    height, width = (
        _normalize_output_shape_component(value) for value in flat_shape.tolist()
    )
    return height, width


def _normalize_output_shape_component(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_OUTPUT_SHAPE_ERROR)

    try:
        integer_value = int(operator.index(value))
    except TypeError:
        integer_value = None

    if integer_value is None:
        if isinstance(value, (float, np.floating)):
            numeric_value = float(value)
            if np.isfinite(numeric_value) and numeric_value.is_integer():
                integer_value = int(numeric_value)
            else:
                raise ValueError(_OUTPUT_SHAPE_ERROR)
        else:
            raise ValueError(_OUTPUT_SHAPE_ERROR)

    if integer_value < 0:
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    return integer_value


def _normalize_affine_matrix_xy(matrix_xy: Any) -> np.ndarray:
    try:
        object_matrix = np.asarray(matrix_xy, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_MATRIX_ERROR) from exc

    if object_matrix.shape != (2, 3):
        raise ValueError(_MATRIX_ERROR)

    for value in object_matrix.reshape(-1).tolist():
        if isinstance(value, (bool, np.bool_, str)):
            raise ValueError(_MATRIX_ERROR)

    try:
        matrix = np.asarray(object_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_MATRIX_ERROR) from exc

    if not np.all(np.isfinite(matrix)):
        raise ValueError(_MATRIX_ERROR)
    return matrix


__all__ = ["install_fov_affine_warp_validation"]
