"""Strict validation for FOV-affine estimator controls."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_affine_estimator_validation"
_SUBTRACT_MEAN_ERROR = "subtract_mean must be a boolean"
_GRID_SHAPE_ERROR = "grid_shape must contain exactly two positive integer dimensions"
_MIN_TILE_SIZE_ERROR = "min_tile_size must be a positive integer"
_MAX_SHIFT_FRACTION_ERROR = "max_shift_fraction must be a finite non-negative scalar"


def install_fov_affine_estimator_validation() -> None:
    """Install idempotent validation for tile-affine estimator options."""

    from . import fov_affine_registration as _fov_affine_registration

    original = _fov_affine_registration.estimate_fov_affine_transform
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def estimate_fov_affine_transform_with_validation(
        reference_fov: Any,
        measurement_fov: Any,
        *,
        subtract_mean: Any = True,
        grid_shape: Any = (3, 3),
        min_tile_size: Any = 32,
        max_shift_fraction: Any = 0.55,
    ) -> Any:
        return original(
            reference_fov,
            measurement_fov,
            subtract_mean=_normalize_bool(subtract_mean, _SUBTRACT_MEAN_ERROR),
            grid_shape=_normalize_grid_shape(grid_shape),
            min_tile_size=_normalize_positive_int(min_tile_size, _MIN_TILE_SIZE_ERROR),
            max_shift_fraction=_normalize_nonnegative_float(max_shift_fraction),
        )

    setattr(estimate_fov_affine_transform_with_validation, _PATCH_MARKER, True)
    setattr(estimate_fov_affine_transform_with_validation, "_bayescatrack_original", original)
    _fov_affine_registration.estimate_fov_affine_transform = estimate_fov_affine_transform_with_validation


def _normalize_bool(value: Any, error_message: str) -> bool:
    if type(value) is not bool:
        raise ValueError(error_message)
    return value


def _normalize_grid_shape(value: Any) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise ValueError(_GRID_SHAPE_ERROR)
    try:
        values = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(_GRID_SHAPE_ERROR) from exc
    if values.size != 2:
        raise ValueError(_GRID_SHAPE_ERROR)
    return tuple(
        _normalize_positive_int(part, _GRID_SHAPE_ERROR) for part in values.tolist()
    )


def _normalize_positive_int(value: Any, error_message: str) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(error_message)
    try:
        result = operator.index(value)
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(error_message) from None
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(error_message)
        result = int(numeric)
    result = int(result)
    if result <= 0:
        raise ValueError(error_message)
    return result


def _normalize_nonnegative_float(value: Any) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(_MAX_SHIFT_FRACTION_ERROR)
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_MAX_SHIFT_FRACTION_ERROR) from exc
    if array.shape != ():
        raise ValueError(_MAX_SHIFT_FRACTION_ERROR)
    try:
        result = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(_MAX_SHIFT_FRACTION_ERROR) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(_MAX_SHIFT_FRACTION_ERROR)
    return result


__all__ = ["install_fov_affine_estimator_validation"]