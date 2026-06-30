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
_ESTIMATE_PATCH_MARKER = "_bayescatrack_fov_affine_estimate_control_validation_patch"
_MATRIX_ERROR = "matrix_xy must be a finite 2-by-3 affine matrix"
_OUTPUT_SHAPE_ERROR = (
    "output_shape must contain exactly two positive integer dimensions"
)
_SUBTRACT_MEAN_ERROR = "subtract_mean must be a boolean"
_GRID_SHAPE_ERROR = "grid_shape must contain exactly two positive integer dimensions"
_MIN_TILE_SIZE_ERROR = "min_tile_size must be a positive integer"
_MAX_SHIFT_FRACTION_ERROR = "max_shift_fraction must be a finite non-negative value"
_TEXT_OR_BYTES_LIKE_TYPES = (str, bytes, bytearray, memoryview)


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

    _patch_estimate_controls(_fov_affine_registration)


def _patch_estimate_controls(module: Any) -> None:
    original_estimate = module.estimate_fov_affine_transform
    if getattr(original_estimate, _ESTIMATE_PATCH_MARKER, False):
        return

    @wraps(original_estimate)
    def estimate_fov_affine_transform_with_validation(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return original_estimate(*args, **_normalize_estimate_kwargs(kwargs))

    _mark_patch(
        estimate_fov_affine_transform_with_validation,
        original_estimate,
        _ESTIMATE_PATCH_MARKER,
    )
    module.estimate_fov_affine_transform = estimate_fov_affine_transform_with_validation


def _mark_patch(wrapper: Any, original: Any, marker: str) -> None:
    setattr(wrapper, marker, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_output_shape_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "output_shape" not in kwargs:
        return kwargs
    normalized_kwargs = dict(kwargs)
    normalized_kwargs["output_shape"] = _normalize_output_shape(kwargs["output_shape"])
    return normalized_kwargs


def _normalize_estimate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized_kwargs = dict(kwargs)
    if "subtract_mean" in normalized_kwargs:
        normalized_kwargs["subtract_mean"] = _normalize_bool(
            normalized_kwargs["subtract_mean"],
            _SUBTRACT_MEAN_ERROR,
        )
    if "grid_shape" in normalized_kwargs:
        normalized_kwargs["grid_shape"] = _normalize_grid_shape(
            normalized_kwargs["grid_shape"]
        )
    if "min_tile_size" in normalized_kwargs:
        normalized_kwargs["min_tile_size"] = _normalize_positive_integer(
            normalized_kwargs["min_tile_size"],
            _MIN_TILE_SIZE_ERROR,
        )
    if "max_shift_fraction" in normalized_kwargs:
        normalized_kwargs["max_shift_fraction"] = _normalize_nonnegative_float(
            normalized_kwargs["max_shift_fraction"],
            _MAX_SHIFT_FRACTION_ERROR,
        )
    return normalized_kwargs


def _normalize_output_shape(output_shape: Any) -> tuple[int, int]:
    if isinstance(output_shape, _TEXT_OR_BYTES_LIKE_TYPES):
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    try:
        shape_array = np.asarray(output_shape, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_OUTPUT_SHAPE_ERROR) from exc

    if shape_array.shape != (2,):
        raise ValueError(_OUTPUT_SHAPE_ERROR)

    height, width = (
        _normalize_output_shape_component(value) for value in shape_array.tolist()
    )
    return height, width


def _normalize_output_shape_component(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or isinstance(
        value, _TEXT_OR_BYTES_LIKE_TYPES
    ):
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

    if integer_value <= 0:
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    return integer_value


def _normalize_bool(value: Any, error_message: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(error_message)


def _normalize_grid_shape(grid_shape: Any) -> tuple[int, int]:
    if isinstance(grid_shape, _TEXT_OR_BYTES_LIKE_TYPES):
        raise ValueError(_GRID_SHAPE_ERROR)
    try:
        shape_array = np.asarray(grid_shape, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_GRID_SHAPE_ERROR) from exc
    if shape_array.shape != (2,):
        raise ValueError(_GRID_SHAPE_ERROR)
    rows, cols = (
        _normalize_positive_integer(value, _GRID_SHAPE_ERROR)
        for value in shape_array.tolist()
    )
    return rows, cols


def _normalize_positive_integer(value: Any, error_message: str) -> int:
    if isinstance(value, (bool, np.bool_)) or isinstance(
        value, _TEXT_OR_BYTES_LIKE_TYPES
    ):
        raise ValueError(error_message)

    try:
        integer_value = int(operator.index(value))
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(error_message) from None
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(error_message)
        integer_value = int(numeric_value)

    if integer_value <= 0:
        raise ValueError(error_message)
    return int(integer_value)


def _normalize_nonnegative_float(value: Any, error_message: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(
        value, _TEXT_OR_BYTES_LIKE_TYPES
    ):
        raise ValueError(error_message)
    try:
        value_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc
    if value_array.shape != ():
        raise ValueError(error_message)

    try:
        numeric_value = float(value_array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc

    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(error_message)
    return numeric_value


def _normalize_affine_matrix_xy(matrix_xy: Any) -> np.ndarray:
    if isinstance(matrix_xy, _TEXT_OR_BYTES_LIKE_TYPES):
        raise ValueError(_MATRIX_ERROR)
    try:
        matrix_array = np.asarray(matrix_xy, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_MATRIX_ERROR) from exc
    if matrix_array.shape != (2, 3):
        raise ValueError(_MATRIX_ERROR)
    if _contains_text_or_bytes_like(matrix_array):
        raise ValueError(_MATRIX_ERROR)
    try:
        numeric_matrix = np.asarray(matrix_array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_MATRIX_ERROR) from exc
    if not np.all(np.isfinite(numeric_matrix)):
        raise ValueError(_MATRIX_ERROR)
    return numeric_matrix


def _contains_text_or_bytes_like(values: np.ndarray) -> bool:
    return any(
        isinstance(value, _TEXT_OR_BYTES_LIKE_TYPES) for value in values.reshape(-1)
    )


__all__ = ["install_fov_affine_warp_validation"]
