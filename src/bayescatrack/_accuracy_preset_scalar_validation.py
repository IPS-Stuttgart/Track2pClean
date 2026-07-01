"""Strict validation for accuracy-preset binary-view scalar controls.

Accuracy presets already reject binary text buffers such as ``bytes`` and
``bytearray`` before falling back to numeric conversion. Python can also coerce
numeric-looking ``memoryview`` objects through ``float(...)``; reject those views
so malformed manifest/config values cannot silently become benchmark controls.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_accuracy_preset_scalar_validation_patch"


def install_accuracy_preset_scalar_validation(accuracy_presets_module: Any) -> None:
    """Install idempotent binary-view guards for accuracy preset scalar helpers."""

    _patch_integer_value(accuracy_presets_module)
    _patch_float_or_none(accuracy_presets_module)


def _patch_integer_value(accuracy_presets_module: Any) -> None:
    original = accuracy_presets_module._integer_value  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _integer_value_without_binary_views(value: Any, *, name: str) -> int:
        if isinstance(value, memoryview):
            raise ValueError(f"{name} must be a positive integer")
        return original(value, name=name)

    _mark_patch(_integer_value_without_binary_views, original)
    accuracy_presets_module._integer_value = (  # pylint: disable=protected-access
        _integer_value_without_binary_views
    )


def _patch_float_or_none(accuracy_presets_module: Any) -> None:
    original = accuracy_presets_module._finite_nonnegative_float_or_none  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _finite_nonnegative_float_or_none_without_binary_views(
        value: Any,
        *,
        name: str,
    ) -> float | None:
        if isinstance(value, memoryview):
            raise ValueError(f"{name} must be a finite non-negative value or None")
        return original(value, name=name)

    _mark_patch(_finite_nonnegative_float_or_none_without_binary_views, original)
    accuracy_presets_module._finite_nonnegative_float_or_none = (  # pylint: disable=protected-access
        _finite_nonnegative_float_or_none_without_binary_views
    )


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


__all__ = ["install_accuracy_preset_scalar_validation"]
