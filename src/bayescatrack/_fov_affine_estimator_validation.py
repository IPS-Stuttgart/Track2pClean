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
_LOW_INFORMATION_FOV_MESSAGES = (
    "constant or empty FOV images",
    "spatial variation for phase-correlation registration",
)
_IDENTITY_AFFINE_XY = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


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
        normalized_subtract_mean = _normalize_bool(subtract_mean, _SUBTRACT_MEAN_ERROR)
        normalized_grid_shape = _normalize_grid_shape(grid_shape)
        normalized_min_tile_size = _normalize_positive_int(
            min_tile_size,
            _MIN_TILE_SIZE_ERROR,
        )
        normalized_max_shift_fraction = _normalize_nonnegative_float(max_shift_fraction)
        if _is_empty_2d_fov(reference_fov) or _is_empty_2d_fov(measurement_fov):
            return _identity_fallback_estimate(_fov_affine_registration)
        try:
            return original(
                reference_fov,
                measurement_fov,
                subtract_mean=normalized_subtract_mean,
                grid_shape=normalized_grid_shape,
                min_tile_size=normalized_min_tile_size,
                max_shift_fraction=normalized_max_shift_fraction,
            )
        except ValueError as exc:
            if not _is_low_information_fov_error(exc):
                raise
            return _identity_fallback_estimate(_fov_affine_registration)

    setattr(estimate_fov_affine_transform_with_validation, _PATCH_MARKER, True)
    setattr(
        estimate_fov_affine_transform_with_validation,
        "_bayescatrack_original",
        original,
    )
    _fov_affine_registration.estimate_fov_affine_transform = (
        estimate_fov_affine_transform_with_validation
    )


def _identity_fallback_estimate(affine_registration_module: Any) -> Any:
    return affine_registration_module.FovAffineEstimate(
        matrix_xy=_IDENTITY_AFFINE_XY.copy(),
        inverse_matrix_xy=affine_registration_module.invert_affine_xy(
            _IDENTITY_AFFINE_XY,
        ),
        tile_reference_xy=np.zeros((0, 2), dtype=float),
        tile_measurement_xy=np.zeros((0, 2), dtype=float),
        tile_shift_yx=np.zeros((1, 2), dtype=float),
        tile_peak_correlation=np.zeros((1,), dtype=float),
        tile_residual_norm=np.zeros((0,), dtype=float),
        fit_rmse=0.0,
        fallback_translation=True,
    )


def _is_low_information_fov_error(exc: ValueError) -> bool:
    message = str(exc)
    return any(fragment in message for fragment in _LOW_INFORMATION_FOV_MESSAGES)


def _is_empty_2d_fov(value: Any) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.ndim == 2 and (array.shape[0] == 0 or array.shape[1] == 0)


def _normalize_bool(value: Any, error_message: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(error_message)


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
