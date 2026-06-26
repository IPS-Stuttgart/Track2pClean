"""Strict validation for growth-analysis categorical identifiers.

Growth analysis treats ROI IDs and session indices as categorical labels. Python
booleans are a subclass of ``int``, and the base growth helpers coerce session
indices with ``int(...)``. Without guards, malformed values such as ``True`` or
``1.5`` can silently select session ``1``.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_OPTIONAL_ROI_PATCH_MARKER = "_bayescatrack_growth_optional_roi_validation_patch"
_SESSION_INDEX_PATCH_MARKER = "_bayescatrack_growth_session_index_validation_patch"


def install_growth_optional_roi_validation() -> None:
    """Install idempotent validation around growth-analysis categorical IDs."""

    from .analysis import growth as _growth  # pylint: disable=import-outside-toplevel

    _install_optional_roi_validation(_growth)
    _install_session_index_validation(_growth)


def _install_optional_roi_validation(_growth: Any) -> None:
    original_optional_roi = _growth._optional_roi
    if getattr(original_optional_roi, _OPTIONAL_ROI_PATCH_MARKER, False):
        return

    @wraps(original_optional_roi)
    def _optional_roi_with_validation(value: object) -> int | None:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("ROI index must be integer-like, got boolean")
        return original_optional_roi(value)

    setattr(_optional_roi_with_validation, _OPTIONAL_ROI_PATCH_MARKER, True)
    setattr(_optional_roi_with_validation, "_bayescatrack_original", original_optional_roi)
    _growth._optional_roi = _optional_roi_with_validation


def _install_session_index_validation(_growth: Any) -> None:
    original_validate_session_index = _growth._validate_session_index
    if getattr(original_validate_session_index, _SESSION_INDEX_PATCH_MARKER, False):
        return

    @wraps(original_validate_session_index)
    def _validate_session_index_with_validation(index: object, n_sessions: int) -> int:
        return original_validate_session_index(
            _session_index_to_int(index),
            n_sessions,
        )

    setattr(_validate_session_index_with_validation, _SESSION_INDEX_PATCH_MARKER, True)
    setattr(
        _validate_session_index_with_validation,
        "_bayescatrack_original",
        original_validate_session_index,
    )
    _growth._validate_session_index = _validate_session_index_with_validation


def _session_index_to_int(index: object) -> int:
    if isinstance(index, (bool, np.bool_)):
        raise ValueError("session index must be integer-like, got boolean")
    if isinstance(index, bytes):
        index = index.decode("utf-8")
    if isinstance(index, (int, np.integer)):
        return int(index)
    if isinstance(index, (float, np.floating)):
        return _parse_integer_like_session_index(float(index), original=index)
    if isinstance(index, str):
        text = index.strip()
        if not text:
            raise ValueError("session index must be integer-like, got empty string")
        try:
            return int(text)
        except ValueError:
            pass
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"session index must be integer-like, got {index!r}") from exc
        return _parse_integer_like_session_index(numeric, original=index)
    raise ValueError(f"session index must be integer-like, got {type(index).__name__}")


def _parse_integer_like_session_index(value: float, *, original: object) -> int:
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"session index must be integer-like, got {original!r}")
    return int(value)


__all__ = ["install_growth_optional_roi_validation"]