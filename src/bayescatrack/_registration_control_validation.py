"""Strict validation for registration-control parameters.

The registration wrappers eventually delegate to PyRecEst's point-set
registration backend.  Without local validation, malformed values such as
``allow_reflection="false"`` are forwarded as truthy objects, while NaN,
negative, or vector-valued numeric controls fail only after entering backend
code.  This module installs an idempotent wrapper so BayesCaTrack rejects these
inputs at the public boundary with deterministic errors.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Literal, cast

import numpy as np

RegistrationModel = Literal["translation", "rigid", "affine"]
_VALID_REGISTRATION_MODELS = {"translation", "rigid", "affine"}
_PATCH_ATTR = "_bayescatrack_registration_control_validation_patch"


# pylint: disable=too-many-arguments


def install_registration_control_validation(registration_module: Any) -> None:
    """Install strict validation around ``register_measurement_plane_to_reference``."""

    original = registration_module.register_measurement_plane_to_reference
    if getattr(original, _PATCH_ATTR, False):
        return

    @wraps(original)
    def register_measurement_plane_to_reference(
        reference_plane: Any,
        measurement_plane: Any,
        *,
        order: str = "xy",
        weighted_centroids: bool = False,
        registration_model: RegistrationModel = "affine",
        registration_max_cost: float | None = None,
        registration_max_iterations: int = 25,
        registration_tolerance: float = 1.0e-8,
        min_matches: int | None = None,
        allow_reflection: bool = False,
        binarize_registered_masks: bool = False,
        registered_mask_threshold: float = 0.5,
    ) -> Any:
        normalized_registration_model = _registration_model(registration_model)
        normalized_registration_max_cost = (
            None
            if registration_max_cost is None
            else _finite_nonnegative_float(
                registration_max_cost,
                name="registration_max_cost",
            )
        )
        normalized_min_matches = (
            None
            if min_matches is None
            else _positive_integer(min_matches, name="min_matches")
        )
        return original(
            reference_plane,
            measurement_plane,
            order=order,
            weighted_centroids=_boolean_flag(
                weighted_centroids,
                name="weighted_centroids",
            ),
            registration_model=normalized_registration_model,
            registration_max_cost=normalized_registration_max_cost,
            registration_max_iterations=_positive_integer(
                registration_max_iterations,
                name="registration_max_iterations",
            ),
            registration_tolerance=_finite_nonnegative_float(
                registration_tolerance,
                name="registration_tolerance",
            ),
            min_matches=normalized_min_matches,
            allow_reflection=_boolean_flag(
                allow_reflection,
                name="allow_reflection",
            ),
            binarize_registered_masks=binarize_registered_masks,
            registered_mask_threshold=registered_mask_threshold,
        )

    setattr(register_measurement_plane_to_reference, _PATCH_ATTR, True)
    setattr(register_measurement_plane_to_reference, "_bayescatrack_original", original)
    registration_module.register_measurement_plane_to_reference = (
        register_measurement_plane_to_reference
    )


def _registration_model(value: Any) -> RegistrationModel:
    if not isinstance(value, str) or value not in _VALID_REGISTRATION_MODELS:
        raise ValueError(
            "registration_model must be one of 'translation', 'rigid', or 'affine'"
        )
    return cast(RegistrationModel, value)


def _boolean_flag(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a finite non-negative scalar")

    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        numeric_value = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return float(numeric_value)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a positive integer")

    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be a positive integer")
        integer_value = int(value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError:
            array = np.asarray(value)
            if array.shape != ():
                raise ValueError(f"{name} must be a positive integer") from None
            if np.issubdtype(array.dtype, np.integer):
                integer_value = int(array)
            elif np.issubdtype(array.dtype, np.floating):
                numeric_value = float(array)
                if not np.isfinite(numeric_value) or not numeric_value.is_integer():
                    raise ValueError(f"{name} must be a positive integer") from None
                integer_value = int(numeric_value)
            else:
                raise ValueError(f"{name} must be a positive integer") from None

    if integer_value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(integer_value)


__all__ = ["install_registration_control_validation"]
