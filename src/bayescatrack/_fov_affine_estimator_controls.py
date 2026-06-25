from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_affine_estimator_controls"


def install_fov_affine_estimator_control_validation() -> None:
    from . import fov_affine_registration as module  # pylint: disable=import-outside-toplevel

    original = module.estimate_fov_affine_transform
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def estimate_fov_affine_transform_checked(
        reference_fov: Any,
        measurement_fov: Any,
        *args: Any,
        grid_shape: Any = (3, 3),
        min_tile_size: Any = 32,
        max_shift_fraction: Any = 0.55,
        **kwargs: Any,
    ) -> Any:
        return original(
            reference_fov,
            measurement_fov,
            *args,
            grid_shape=_normalize_grid_shape(grid_shape),
            min_tile_size=_normalize_positive_int(min_tile_size, "min_tile_size"),
            max_shift_fraction=_normalize_nonnegative_float(
                max_shift_fraction,
                "max_shift_fraction",
            ),
            **kwargs,
        )

    setattr(estimate_fov_affine_transform_checked, _PATCH_MARKER, True)
    setattr(estimate_fov_affine_transform_checked, "_bayescatrack_original", original)
    module.estimate_fov_affine_transform = estimate_fov_affine_transform_checked


def _normalize_grid_shape(value: Any) -> tuple[int, int]:
    message = "grid_shape must contain exactly two positive integer dimensions"
    if isinstance(value, (str, bytes)):
        raise ValueError(message)
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != (2,):
        raise ValueError(message)
    rows, cols = (_normalize_positive_int(part, "grid_shape") for part in array.tolist())
    return rows, cols


def _normalize_positive_int(value: Any, name: str) -> int:
    message = (
        "grid_shape must contain exactly two positive integer dimensions"
        if name == "grid_shape"
        else f"{name} must be a positive integer"
    )
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(message)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(message)
        normalized = int(numeric)
    else:
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise ValueError(message) from exc
    normalized = int(normalized)
    if normalized <= 0:
        raise ValueError(message)
    return normalized


def _normalize_nonnegative_float(value: Any, name: str) -> float:
    message = f"{name} must be a finite non-negative value"
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(message)
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != ():
        raise ValueError(message)
    try:
        normalized = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(message)
    return normalized


__all__ = ["install_fov_affine_estimator_control_validation"]
