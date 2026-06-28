"""Validation patch for nonrigid-registration runtime controls."""

from __future__ import annotations

import sys
from numbers import Integral
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_nonrigid_registration_control_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_nonrigid_registration_control_validation_original"


def install_nonrigid_registration_control_validation() -> None:
    """Reject malformed nonrigid-registration controls before warp fitting."""

    from . import nonrigid_registration as _nonrigid_registration

    if getattr(_nonrigid_registration, _PATCH_MARKER, False):
        return

    original = _nonrigid_registration.register_measurement_plane_by_nonrigid_fov

    def validated_register_measurement_plane_by_nonrigid_fov(
        reference_plane: Any,
        measurement_plane: Any,
        *,
        transform_type: Any = "bspline",
        grid_shape: tuple[int, int] = (5, 5),
        min_tile_size: int = 24,
        max_shift_fraction: float = 0.75,
        tps_regularization: Any = 1.0e-3,
        bspline_regularization: Any = 1.0e-2,
        optical_flow_iterations: Any = 12,
        optical_flow_alpha: Any = 25.0,
        **unused_options: object,
    ) -> Any:
        if unused_options:
            return original(
                reference_plane,
                measurement_plane,
                transform_type=transform_type,
                grid_shape=grid_shape,
                min_tile_size=min_tile_size,
                max_shift_fraction=max_shift_fraction,
                tps_regularization=tps_regularization,
                bspline_regularization=bspline_regularization,
                optical_flow_iterations=optical_flow_iterations,
                optical_flow_alpha=optical_flow_alpha,
                **unused_options,
            )

        return original(
            reference_plane,
            measurement_plane,
            transform_type=transform_type,
            grid_shape=grid_shape,
            min_tile_size=min_tile_size,
            max_shift_fraction=max_shift_fraction,
            tps_regularization=_finite_nonnegative_float(
                tps_regularization,
                name="tps_regularization",
            ),
            bspline_regularization=_finite_nonnegative_float(
                bspline_regularization,
                name="bspline_regularization",
            ),
            optical_flow_iterations=_nonnegative_integer(
                optical_flow_iterations,
                name="optical_flow_iterations",
            ),
            optical_flow_alpha=_finite_positive_float(
                optical_flow_alpha,
                name="optical_flow_alpha",
            ),
        )

    validated_register_measurement_plane_by_nonrigid_fov.__name__ = original.__name__
    validated_register_measurement_plane_by_nonrigid_fov.__qualname__ = (
        original.__qualname__
    )
    setattr(
        validated_register_measurement_plane_by_nonrigid_fov, _ORIGINAL_ATTR, original
    )
    _nonrigid_registration.register_measurement_plane_by_nonrigid_fov = (
        validated_register_measurement_plane_by_nonrigid_fov
    )
    track2p_registration = sys.modules.get(f"{__package__}.track2p_registration")
    if track2p_registration is not None:
        setattr(
            track2p_registration,
            "register_measurement_plane_by_nonrigid_fov",
            validated_register_measurement_plane_by_nonrigid_fov,
        )
    setattr(_nonrigid_registration, _PATCH_MARKER, True)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    error_message = f"{name} must be finite and non-negative"
    numeric_value = _finite_float(value, error_message=error_message)
    if numeric_value < 0.0:
        raise ValueError(error_message)
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    error_message = f"{name} must be finite and strictly positive"
    numeric_value = _finite_float(value, error_message=error_message)
    if numeric_value <= 0.0:
        raise ValueError(error_message)
    return numeric_value


def _finite_float(value: Any, *, error_message: str) -> float:
    scalar = _scalar_value(value, error_message=error_message)
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(error_message)
    try:
        numeric_value = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(error_message)
    return numeric_value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    error_message = f"{name} must be a non-negative integer"
    scalar = _scalar_value(value, error_message=error_message)
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Integral):
        raise ValueError(error_message)
    integer_value = int(scalar)
    if integer_value < 0:
        raise ValueError(error_message)
    return integer_value


def _scalar_value(value: Any, *, error_message: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(error_message)
    return array.item()


__all__ = ["install_nonrigid_registration_control_validation"]
