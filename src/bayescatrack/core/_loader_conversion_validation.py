"""Normalize Suite2p loader numeric-control conversion failures."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_loader_numeric_conversion_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_ERROR_TEMPLATE = "{name} must be a finite probability"
_STRING_LIKE_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_TARGET_NAME = "_" + "finite_probability"


def install_loader_numeric_conversion_validation(loader_validation_module: ModuleType) -> None:
    """Wrap loader probability conversion errors in public ValueError messages."""

    original_probability = getattr(loader_validation_module, _TARGET_NAME)
    if getattr(original_probability, _PATCH_MARKER, False):
        return

    def probability(value: Any, *, name: str) -> float:
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

    setattr(probability, _PATCH_MARKER, True)
    setattr(probability, _ORIGINAL_ATTR, original_probability)
    setattr(loader_validation_module, _TARGET_NAME, probability)


__all__ = ["install_loader_numeric_conversion_validation"]
