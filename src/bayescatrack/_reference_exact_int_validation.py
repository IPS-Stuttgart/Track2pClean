"""Exact textual ROI-index parsing for Track2p reference helpers.

Reference tables can arrive from CSV/NumPy object arrays with integer ROI IDs
encoded as text. The base parser historically accepted decimal strings such as
``"7.0"`` by going through ``float(...)``. That silently rounds integer-valued
strings above the IEEE-754 exact-integer range, which can turn a valid ROI ID
into a neighboring one before scoring.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from types import ModuleType
from typing import Any

_PATCH_ATTR = "_bayescatrack_reference_exact_int_validation_patch"


def install_reference_exact_int_validation(reference_module: ModuleType | None = None) -> None:
    """Install idempotent exact parsing for textual reference ROI indices."""

    if reference_module is None:
        from . import reference as reference_module  # pylint: disable=import-outside-toplevel,reimported

    original_parse_optional_int = reference_module._parse_optional_int  # pylint: disable=protected-access
    if getattr(original_parse_optional_int, _PATCH_ATTR, False):
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

    setattr(_parse_optional_int_with_exact_text, _PATCH_ATTR, True)
    setattr(_parse_optional_int_with_exact_text, "_bayescatrack_original", original_parse_optional_int)
    reference_module._parse_optional_int = _parse_optional_int_with_exact_text  # pylint: disable=protected-access


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
