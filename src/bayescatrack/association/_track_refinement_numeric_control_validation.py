"""Strict validation for track-refinement numeric control scalars.

``TrackSmoothingConfig`` thresholds and detection-count controls are scalar
controls, but the underlying implementation normalizes them with permissive
``float(...)`` and ``operator.index(...)`` coercions. NumPy currently allows
``float(np.array([3.5]))`` and ``float(np.array([True]))`` with only a warning,
and custom integer-like objects can raise low-level index-protocol exceptions.
Validate these controls before that coercion happens.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_refinement_numeric_control_validation_patch"


def install_track_refinement_numeric_control_validation() -> None:
    """Install idempotent validation for track-refinement numeric controls."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original_post_init = module.TrackSmoothingConfig.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def checked_post_init(self: Any) -> None:
        residual_z_threshold = _normalize_float_control(
            self.residual_z_threshold,
            name="residual_z_threshold",
            allow_zero=False,
        )
        min_track_detections = _normalize_integer_control(
            self.min_track_detections,
            name="min_track_detections",
            minimum=2,
        )
        min_edge_residual = _normalize_float_control(
            self.min_edge_residual,
            name="min_edge_residual",
            allow_zero=True,
        )
        object.__setattr__(self, "min_track_detections", min_track_detections)
        original_post_init(self)
        object.__setattr__(self, "residual_z_threshold", residual_z_threshold)
        object.__setattr__(self, "min_track_detections", min_track_detections)
        object.__setattr__(self, "min_edge_residual", min_edge_residual)

    setattr(checked_post_init, _PATCH_MARKER, True)
    setattr(checked_post_init, "_bayescatrack_original", original_post_init)
    module.TrackSmoothingConfig.__post_init__ = checked_post_init


def _normalize_float_control(value: Any, *, name: str, allow_zero: bool) -> float:
    bound_description = "finite non-negative" if allow_zero else "finite positive"
    message = f"{name} must be a {bound_description} value"

    if isinstance(value, (bool, np.bool_, bytes, bytearray, np.bytes_)):
        raise ValueError(message)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(message)
        try:
            normalized = float(stripped)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
    else:
        try:
            value_array = np.asarray(value, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if value_array.shape != ():
            raise ValueError(message)
        scalar = value_array.item()
        if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray, np.bytes_)):
            raise ValueError(message)
        try:
            normalized = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc

    if not np.isfinite(normalized):
        raise ValueError(message)
    if allow_zero:
        if normalized < 0.0:
            raise ValueError(message)
    elif normalized <= 0.0:
        raise ValueError(message)
    return normalized


def _normalize_integer_control(value: Any, *, name: str, minimum: int) -> int:
    integer_message = f"{name} must be an integer"
    minimum_message = f"{name} must be at least {minimum}"

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(integer_message)

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(integer_message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(integer_message)

    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
    elif isinstance(value, (float, np.floating)):
        integer_value = _integer_from_float(float(value), message=integer_message)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(integer_message)
        try:
            integer_value = _integer_from_float(
                float(stripped),
                message=integer_message,
            )
        except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
            raise ValueError(integer_message) from exc
    elif isinstance(value, (bytes, bytearray, np.bytes_)):
        raise ValueError(integer_message)
    else:
        try:
            integer_value = int(operator.index(value))
        except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
            raise ValueError(integer_message) from exc

    if integer_value < minimum:
        raise ValueError(minimum_message)
    return integer_value


def _integer_from_float(value: float, *, message: str) -> int:
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(message)
    return int(value)


__all__ = ["install_track_refinement_numeric_control_validation"]
