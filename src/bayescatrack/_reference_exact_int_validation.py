"""Exact ROI-index parsing for Track2p reference helpers.

Reference tables can arrive from CSV/NumPy object arrays with integer ROI IDs
encoded as text, but programmatic benchmark helpers also receive Python and NumPy
integer scalars.  The base parser historically accepted decimal strings and
non-text scalars by going through ``float(...)``. That silently rounds
integer-valued values above the IEEE-754 exact-integer range, which can turn a
valid ROI ID into a neighboring one before scoring.
"""

from __future__ import annotations

import operator
from decimal import Decimal, InvalidOperation
from types import ModuleType
from typing import Any

import numpy as np

_OPTIONAL_PATCH_ATTR = "_bayescatrack_reference_exact_int_validation_patch"
_SCALAR_PATCH_ATTR = "_bayescatrack_reference_exact_scalar_validation_patch"
_PLATFORM_INT_MIN = int(np.iinfo(np.intp).min)
_PLATFORM_INT_MAX = int(np.iinfo(np.intp).max)


def install_reference_exact_int_validation(reference_module: ModuleType | None = None) -> None:
    """Install idempotent exact parsing for reference ROI/index scalars."""

    if reference_module is None:
        from . import reference as reference_module  # pylint: disable=import-outside-toplevel,reimported

    _install_parse_integer_scalar_exact(reference_module)

    original_parse_optional_int = reference_module._parse_optional_int  # pylint: disable=protected-access
    if getattr(original_parse_optional_int, _OPTIONAL_PATCH_ATTR, False):
        return

    missing_strings = frozenset(reference_module._MISSING_STRINGS)  # pylint: disable=protected-access
    error_message = reference_module._optional_int_error_message  # pylint: disable=protected-access

    def _parse_optional_int_with_exact_text(value: Any) -> int | None:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(error_message(value)) from exc
        if isinstance(value, str):
            return _parse_textual_integer_like_roi(
                value,
                missing_strings=missing_strings,
                error_message=error_message,
            )
        return original_parse_optional_int(value)

    setattr(_parse_optional_int_with_exact_text, _OPTIONAL_PATCH_ATTR, True)
    setattr(
        _parse_optional_int_with_exact_text,
        "_bayescatrack_original",
        original_parse_optional_int,
    )
    reference_module._parse_optional_int = _parse_optional_int_with_exact_text  # pylint: disable=protected-access


def _install_parse_integer_scalar_exact(reference_module: ModuleType) -> None:
    original_parse_integer_scalar = reference_module._parse_integer_scalar  # pylint: disable=protected-access
    if getattr(original_parse_integer_scalar, _SCALAR_PATCH_ATTR, False):
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

    setattr(_parse_integer_scalar_with_exact_validation, _SCALAR_PATCH_ATTR, True)
    setattr(
        _parse_integer_scalar_with_exact_validation,
        "_bayescatrack_original",
        original_parse_integer_scalar,
    )
    reference_module._parse_integer_scalar = _parse_integer_scalar_with_exact_validation  # pylint: disable=protected-access


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
            return _parse_decimal_integer(scalar, error_message=lambda _text: message)
    try:
        return int(operator.index(scalar))
    except (TypeError, ValueError, OverflowError):
        numeric_value = _finite_float(scalar, message=message)
        if not numeric_value.is_integer():
            raise ValueError(message)
        return int(numeric_value)


def _finite_float(value: Any, *, message: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(message)
    return numeric_value


def _parse_textual_integer_like_roi(
    value: str,
    *,
    missing_strings: frozenset[str],
    error_message: Any,
) -> int | None:
    text = value.strip()
    if text.lower() in missing_strings:
        return None

    try:
        integer_value = int(text, 10)
    except ValueError:
        integer_value = _parse_decimal_integer(text, error_message=error_message)

    if integer_value < 0:
        return None
    return integer_value


def _parse_decimal_integer(text: str, *, error_message: Any) -> int:
    try:
        numeric_value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(error_message(text)) from exc
    if numeric_value.is_nan():
        return -1
    if not numeric_value.is_finite() or numeric_value != numeric_value.to_integral_value():
        raise ValueError(error_message(text))
    return int(numeric_value)


__all__ = ["install_reference_exact_int_validation"]
