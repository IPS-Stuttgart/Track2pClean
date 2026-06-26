"""Strict validation for growth-analysis track-table ROI identifiers.

Growth analysis treats ROI IDs as categorical labels. Python booleans are a
subclass of ``int``, so accepting them through the integer branch can silently
turn malformed track-table entries into ROI ``0`` or ``1``.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCH_MARKER = "_bayescatrack_growth_optional_roi_validation_patch"


def install_growth_optional_roi_validation() -> None:
    """Install idempotent validation around growth-analysis optional ROI parsing."""

    from .analysis import growth as _growth  # pylint: disable=import-outside-toplevel

    original_optional_roi = _growth._optional_roi
    if getattr(original_optional_roi, _PATCH_MARKER, False):
        return

    @wraps(original_optional_roi)
    def _optional_roi_with_validation(value: object) -> int | None:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("ROI index must be integer-like, got boolean")
        return original_optional_roi(value)

    setattr(_optional_roi_with_validation, _PATCH_MARKER, True)
    setattr(_optional_roi_with_validation, "_bayescatrack_original", original_optional_roi)
    _growth._optional_roi = _optional_roi_with_validation


__all__ = ["install_growth_optional_roi_validation"]
