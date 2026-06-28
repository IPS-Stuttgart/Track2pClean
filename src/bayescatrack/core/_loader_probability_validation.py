"""Validation patch for Suite2p loader probability controls."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_loader_probability_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_ERROR_TEMPLATE = "{name} must be a finite probability"
_STRING_LIKE_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_loader_probability_validation(loader_validation_module: ModuleType) -> None:
    """Reject ambiguous loader probability controls before loader patch install."""

    original_finite_probability = loader_validation_module._finite_probability  # pylint: disable=protected-access
    if getattr(original_finite_probability, _PATCH_MARKER, False):
        return

    def _finite_probability(value: Any, *, name: str) -> float:
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
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(_ERROR_TEMPLATE.format(name=name)) from exc
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(_ERROR_TEMPLATE.format(name=name))
        return numeric

    setattr(_finite_probability, _PATCH_MARKER, True)
    setattr(_finite_probability, _ORIGINAL_ATTR, original_finite_probability)
    loader_validation_module._finite_probability = _finite_probability  # pylint: disable=protected-access


__all__ = ["install_loader_probability_validation"]