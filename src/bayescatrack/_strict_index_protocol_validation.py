"""Normalize strict integer-control index protocol failures."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_PATCH_MARKER = "_bayescatrack_strict_index_protocol_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"


def install_strict_index_protocol_validation() -> None:
    """Normalize malformed ``__index__`` failures in strict integer controls."""

    from . import _strict_config_validation as strict  # pylint: disable=import-outside-toplevel

    original = strict._positive_int  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def positive_int_with_index_error_normalization(value: Any, *, name: str) -> int:
        try:
            return original(value, name=name)
        except ValueError as exc:
            if _is_existing_strict_integer_error(exc, name=name):
                raise
            raise ValueError(f"{name} must be an integer") from exc
        except ArithmeticError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    setattr(positive_int_with_index_error_normalization, _PATCH_MARKER, True)
    setattr(positive_int_with_index_error_normalization, _ORIGINAL_ATTR, original)
    strict._positive_int = positive_int_with_index_error_normalization  # pylint: disable=protected-access


def _is_existing_strict_integer_error(exc: ValueError, *, name: str) -> bool:
    return str(exc).startswith(f"{name} must be")


__all__ = ["install_strict_index_protocol_validation"]
