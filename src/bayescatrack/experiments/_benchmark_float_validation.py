"""Strict validation for benchmark-manifest floating-point options."""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_benchmark_float_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_REJECTED_FLOAT_SCALAR_TYPES = (
    bool,
    np.bool_,
    bytes,
    bytearray,
    memoryview,
    np.bytes_,
)
_NONE_FLOAT_STRINGS = {"", "none", "null", "off", "disabled"}


def install_benchmark_float_validation() -> None:
    """Install idempotent validation for benchmark manifest float controls."""

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        benchmark_manifest as manifest,
    )

    current_float = manifest._float_option  # pylint: disable=protected-access
    if not _callable_chain_has_patch(current_float):
        manifest._float_option = _wrap_float_option(  # pylint: disable=protected-access
            current_float
        )

    current_optional_float = (
        manifest._optional_float_option  # pylint: disable=protected-access
    )
    if not _callable_chain_has_patch(current_optional_float):
        manifest._optional_float_option = (  # pylint: disable=protected-access
            _wrap_optional_float_option(current_optional_float)
        )


def _wrap_float_option(original: Callable[..., float]) -> Callable[..., float]:
    @wraps(original)
    def float_option_with_strict_scalar_validation(
        options: Any, key: str, *, default: float
    ) -> float:
        return _finite_float_scalar(options.get(key, default), name=key)

    setattr(float_option_with_strict_scalar_validation, _PATCH_MARKER, True)
    setattr(float_option_with_strict_scalar_validation, _ORIGINAL_ATTR, original)
    return float_option_with_strict_scalar_validation


def _wrap_optional_float_option(
    original: Callable[..., float | None],
) -> Callable[..., float | None]:
    @wraps(original)
    def optional_float_option_with_strict_scalar_validation(
        options: Any, *keys: str
    ) -> float | None:
        for key in keys:
            if key not in options or options[key] is None:
                continue
            value = options[key]
            if (
                isinstance(value, str)
                and value.strip().casefold() in _NONE_FLOAT_STRINGS
            ):
                return None
            return _finite_float_scalar(value, name=key)
        return None

    setattr(optional_float_option_with_strict_scalar_validation, _PATCH_MARKER, True)
    setattr(
        optional_float_option_with_strict_scalar_validation,
        _ORIGINAL_ATTR,
        original,
    )
    return optional_float_option_with_strict_scalar_validation


def _finite_float_scalar(value: Any, *, name: str) -> float:
    error_message = f"{name} must be a finite numeric scalar"
    if isinstance(value, _REJECTED_FLOAT_SCALAR_TYPES):
        raise ValueError(error_message)

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(error_message)
        value = value.item()
        if isinstance(value, _REJECTED_FLOAT_SCALAR_TYPES):
            raise ValueError(error_message)

    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
        raise ValueError(error_message) from exc

    if not math.isfinite(normalized):
        raise ValueError(error_message)
    return normalized


def _callable_chain_has_patch(function: Any) -> bool:
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


__all__ = ["install_benchmark_float_validation"]
