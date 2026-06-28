"""Normalize absence-model ROI-count numeric conversion failures.

The absence model accepts integer-like ``plane.n_rois`` values at public helper
boundaries.  The base validator already rejects booleans, non-scalars,
non-finite values, fractional values, and negative counts, but custom numeric
adapters can raise their own ``ValueError``/``OverflowError`` while the
validator probes ``operator.index``.  Surface those failures through the same
BayesCaTrack validation message as the ordinary bad-count paths.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

_PATCH_ATTR = "_track2pclean_absence_roi_count_validation"
_ORIGINAL_ATTR = "_track2pclean_absence_roi_count_validation_original"


def install_absence_roi_count_validation(absence_model: ModuleType) -> None:
    """Install idempotent normalization for absence-model ROI-count controls."""

    original_validator = absence_model._validated_roi_count
    if getattr(original_validator, _PATCH_ATTR, False):
        return

    @wraps(original_validator)
    def _validated_roi_count(plane: Any, plane_name: str) -> int:
        try:
            return original_validator(plane, plane_name)
        except (ValueError, OverflowError) as exc:
            raise ValueError(_message(plane_name)) from exc

    setattr(_validated_roi_count, _PATCH_ATTR, True)
    setattr(_validated_roi_count, _ORIGINAL_ATTR, original_validator)
    absence_model._validated_roi_count = _validated_roi_count


def _message(plane_name: str) -> str:
    return f"{plane_name}.n_rois must be a finite non-negative integer"


__all__ = ["install_absence_roi_count_validation"]
