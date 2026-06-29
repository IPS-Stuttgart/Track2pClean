"""Overflow-safe registration-control validation patch."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_registration_control_overflow_validation_patch"
_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_registration_control_overflow_validation() -> None:
    """Normalize overflow failures from registration-control coercion."""

    from . import registration as _registration  # pylint: disable=import-outside-toplevel

    original = _registration.register_measurement_plane_to_reference
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def register_measurement_plane_to_reference_with_overflow_validation(
        reference_plane: Any,
        measurement_plane: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("registration_max_cost") is not None:
            _validate_nonnegative_float_control(
                kwargs["registration_max_cost"],
                name="registration_max_cost",
            )
        if "registration_tolerance" in kwargs:
            _validate_nonnegative_float_control(
                kwargs["registration_tolerance"],
                name="registration_tolerance",
            )
        if "registration_max_iterations" in kwargs:
            _validate_positive_integer_control(
                kwargs["registration_max_iterations"],
                name="registration_max_iterations",
            )
        if kwargs.get("min_matches") is not None:
            _validate_positive_integer_control(
                kwargs["min_matches"],
                name="min_matches",
            )
        return original(reference_plane, measurement_plane, *args, **kwargs)

    setattr(
        register_measurement_plane_to_reference_with_overflow_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        register_measurement_plane_to_reference_with_overflow_validation,
        "_bayescatrack_original",
        original,
    )
    _registration.register_measurement_plane_to_reference = (
        register_measurement_plane_to_reference_with_overflow_validation
    )


def _validate_nonnegative_float_control(value: Any, *, name: str) -> None:
    message = f"{name} must be a finite non-negative scalar"
    if isinstance(value, (bool, np.bool_, *_TEXT_TYPES)):
        raise ValueError(message)
    try:
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError(message)
        numeric_value = float(value_array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(message)


def _validate_positive_integer_control(value: Any, *, name: str) -> None:
    message = f"{name} must be a positive integer"
    if isinstance(value, (bool, np.bool_, *_TEXT_TYPES)):
        raise ValueError(message)
    try:
        integer_value = _coerce_integer(value, message=message)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if integer_value < 1:
        raise ValueError(message)


def _coerce_integer(value: Any, *, message: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(message)
        return int(value)
    try:
        return int(operator.index(value))
    except TypeError:
        value_array = np.asarray(value)
    if value_array.shape != ():
        raise ValueError(message)
    if np.issubdtype(value_array.dtype, np.integer):
        return int(value_array)
    if np.issubdtype(value_array.dtype, np.floating):
        numeric_value = float(value_array)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(message)
        return int(numeric_value)
    raise ValueError(message)


__all__ = ["install_registration_control_overflow_validation"]
