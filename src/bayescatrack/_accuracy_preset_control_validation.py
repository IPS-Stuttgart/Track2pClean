"""Strict validation patches for accuracy-preset scalar controls."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_accuracy_preset_control_validation_patch"


def install_accuracy_preset_control_validation(module: Any) -> None:
    """Patch accuracy-preset helper validators with stable public errors."""

    if not getattr(getattr(module, "_integer_value", None), _PATCH_MARKER, False):
        module._integer_value = _integer_value
    if not getattr(
        getattr(module, "_finite_nonnegative_float_or_none", None),
        _PATCH_MARKER,
        False,
    ):
        module._finite_nonnegative_float_or_none = _finite_nonnegative_float_or_none


def _integer_value(value: Any, *, name: str) -> int:
    message = f"{name} must be a positive integer"
    if isinstance(value, (bool, np.bool_, bytes, bytearray, np.ndarray)):
        raise ValueError(message)
    try:
        return int(operator.index(value))
    except TypeError:
        pass
    except (ValueError, OverflowError) as exc:
        raise ValueError(message) from exc

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError(message)
    else:
        candidate = value
    try:
        numeric_value = float(candidate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(message)
    return int(numeric_value)


def _finite_nonnegative_float_or_none(value: Any, *, name: str) -> float | None:
    message = f"{name} must be a finite non-negative value or None"
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, bytes, bytearray, np.ndarray)):
        raise ValueError(message)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(message)
    return numeric_value


setattr(_integer_value, _PATCH_MARKER, True)
setattr(_finite_nonnegative_float_or_none, _PATCH_MARKER, True)


__all__ = ["install_accuracy_preset_control_validation"]
