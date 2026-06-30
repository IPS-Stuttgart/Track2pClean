"""Validation patch for Suite2p loader probability controls."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_loader_probability_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_ERROR_TEMPLATE = "{name} must be a finite probability"
_STRING_LIKE_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_TARGET_NAME = "_" + "finite_probability"


def install_loader_probability_validation(loader_validation_module: ModuleType) -> None:
    """Reject ambiguous loader probability controls before loader patch install."""

    original_finite_probability = getattr(loader_validation_module, _TARGET_NAME)
    if getattr(original_finite_probability, _PATCH_MARKER, False):
        return

    def replacement(value: Any, *, name: str) -> float:
        if isinstance(value, np.ndarray):
            if value.shape != ():
                raise ValueError(_ERROR_TEMPLATE.format(name=name))
            value = value.item()
        if isinstance(value, (bool, np.bool_)) or isinstance(
            value, _STRING_LIKE_SCALAR_TYPES
        ):
            raise ValueError(_ERROR_TEMPLATE.format(name=name))
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
            raise ValueError(_ERROR_TEMPLATE.format(name=name)) from exc
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(_ERROR_TEMPLATE.format(name=name))
        return numeric

    setattr(replacement, _PATCH_MARKER, True)
    setattr(replacement, _ORIGINAL_ATTR, original_finite_probability)
    setattr(loader_validation_module, _TARGET_NAME, replacement)


__all__ = ["install_loader_probability_validation"]
