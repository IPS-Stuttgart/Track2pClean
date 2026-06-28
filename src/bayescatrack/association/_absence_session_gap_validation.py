"""Normalize absence-model session-gap numeric conversion failures.

``gap_penalty_matrix`` accepts integer-like session-gap offsets, including
NumPy scalar values and numeric strings.  The underlying normalization already
rejects non-finite, fractional, boolean, and non-scalar inputs, but exotic
numeric adapters can raise ``OverflowError`` while being converted with
``float(...)``.  Surface those failures through the same ``ValueError`` contract
as the rest of the session-gap validation path.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

_PATCH_ATTR = "_track2pclean_absence_session_gap_validation"
_ORIGINAL_ATTR = "_track2pclean_absence_session_gap_validation_original"
_ERROR_MESSAGE = (
    "session_gap must be a finite value representing an integer "
    "greater than or equal to 1"
)


def install_absence_session_gap_validation(absence_model: ModuleType) -> None:
    """Install idempotent overflow normalization for session-gap controls."""

    original_validator = absence_model._validated_positive_integer_session_gap
    if getattr(original_validator, _PATCH_ATTR, False):
        return

    @wraps(original_validator)
    def _validated_positive_integer_session_gap(session_gap: Any) -> int:
        try:
            return original_validator(session_gap)
        except OverflowError as exc:
            raise ValueError(_ERROR_MESSAGE) from exc

    setattr(_validated_positive_integer_session_gap, _PATCH_ATTR, True)
    setattr(_validated_positive_integer_session_gap, _ORIGINAL_ATTR, original_validator)
    absence_model._validated_positive_integer_session_gap = (
        _validated_positive_integer_session_gap
    )


__all__ = ["install_absence_session_gap_validation"]
