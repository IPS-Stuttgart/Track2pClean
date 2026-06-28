"""Validation patch for malformed advanced-uncertainty scalar controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import advanced_uncertainty as _advanced_uncertainty

# pylint: disable=protected-access

_PATCH_MARKER = "_bayescatrack_advanced_uncertainty_array_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_advanced_uncertainty_array_validation_original"
_STRING_LIKE_SCALAR_TYPES = (str, bytes, bytearray)


def install_advanced_uncertainty_array_validation() -> None:
    """Require numeric uncertainty/pruning controls to be numeric scalars."""

    original = _advanced_uncertainty._validated_float
    if _method_chain_has_patch(original):
        return

    def validated_float(value: Any, *, name: str) -> float:
        if isinstance(value, _STRING_LIKE_SCALAR_TYPES):
            raise ValueError(f"{name} must be a numeric scalar")
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = value_array.item()
        if isinstance(scalar, _STRING_LIKE_SCALAR_TYPES):
            raise ValueError(f"{name} must be a numeric scalar")
        return original(scalar, name=name)

    validated_float.__name__ = original.__name__
    validated_float.__qualname__ = original.__qualname__
    setattr(validated_float, _PATCH_MARKER, True)
    setattr(validated_float, _ORIGINAL_ATTR, original)
    _advanced_uncertainty._validated_float = validated_float


def _method_chain_has_patch(method: Any) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


__all__ = ["install_advanced_uncertainty_array_validation"]
