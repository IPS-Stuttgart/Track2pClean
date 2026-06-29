"""Exact scalar validation for Track2p reference integer controls."""

from __future__ import annotations

import operator
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_reference_scalar_validation_patch"
_PLATFORM_INT_MIN = int(np.iinfo(np.intp).min)
_PLATFORM_INT_MAX = int(np.iinfo(np.intp).max)


def install_reference_scalar_validation(
    reference_module: ModuleType | None = None,
) -> None:
    """Install an exact, range-checked parser for reference integer scalars."""

    if reference_module is None:
        from . import (  # pylint: disable=import-outside-toplevel,reimported
            reference as reference_module,
        )

    original_parse_integer_scalar = (
        reference_module._parse_integer_scalar  # pylint: disable=protected-access
    )
    if getattr(original_parse_integer_scalar, _PATCH_ATTR, False):
        return

    def _parse_integer_scalar_with_exact_validation(
        value: Any,
        *,
        name: str,
        allow_negative: bool,
        allow_string: bool,
    ) -> int:
        return _parse_integer_scalar_exact(
            value,
            name=name,
            allow_negative=allow_negative,
            allow_string=allow_string,
        )

    setattr(_parse_integer_scalar_with_exact_validation, _PATCH_ATTR, True)
    setattr(
        _parse_integer_scalar_with_exact_validation,
        "_bayescatrack_original",
        original_parse_integer_scalar,
    )
    reference_module._parse_integer_scalar = (  # pylint: disable=protected-access
        _parse_integer_scalar_with_exact_validation
    )


def _parse_integer_scalar_exact(
    value: Any,
    *,
    name: str,
    allow_negative: bool,
    allow_string: bool,
) -> int:
    message = f"{name} must be an integer scalar"
    scalar = _unwrap_scalar(value, message=message, allow_string=allow_string)
    integer_value = _coerce_scalar_to_integer(
        scalar,
        message=message,
        allow_string=allow_string,
    )
    if integer_value < _PLATFORM_INT_MIN or integer_value > _PLATFORM_INT_MAX:
        raise ValueError(message)
    if not allow_negative and integer_value < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return integer_value


def _unwrap_scalar(value: Any, *, message: str, allow_string: bool) -> Any:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, bytes):
        if not allow_string:
            raise ValueError(message)
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(message) from exc
    if isinstance(value, str):
        if not allow_string:
            raise ValueError(message)
        value = value.strip()
    array = np.asarray(value, dtype=object)
    if array.shape != ():
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, bytes):
        if not allow_string:
            raise ValueError(message)
        try:
            return scalar.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(message) from exc
    if isinstance(scalar, str):
        if not allow_string:
            raise ValueError(message)
        return scalar.strip()
    return scalar


def _coerce_scalar_to_integer(
    scalar: Any,
    *,
    message: str,
    allow_string: bool,
) -> int:
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, str):
        if not allow_string:
            raise ValueError(message)
        try:
            return int(scalar, 10)
        except ValueError:
            return _coerce_float_to_integer(
                _finite_float(scalar, message=message),
                message=message,
            )
    try:
        return int(operator.index(scalar))
    except (TypeError, ValueError, OverflowError):
        return _coerce_float_to_integer(
            _finite_float(scalar, message=message),
            message=message,
        )


def _finite_float(value: Any, *, message: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(message)
    return numeric_value


def _coerce_float_to_integer(value: float, *, message: str) -> int:
    if not value.is_integer():
        raise ValueError(message)
    return int(value)


__all__ = ["install_reference_scalar_validation"]
