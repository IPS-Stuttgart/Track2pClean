"""Strict validation for growth-analysis session-index selectors.

Growth analysis accepts source and target session indices through its public
Python API.  Python and NumPy booleans are scalar enough for ``int()`` and can
therefore silently select sessions ``1`` or ``0`` instead of surfacing malformed
configuration.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_growth_session_index_validation_patch"


def install_growth_session_index_validation() -> None:
    """Install idempotent boolean rejection for growth-analysis session indices."""

    from .analysis import growth as _growth  # pylint: disable=import-outside-toplevel

    original_validate_session_index = _growth._validate_session_index
    if getattr(original_validate_session_index, _PATCH_MARKER, False):
        return

    @wraps(original_validate_session_index)
    def _validate_session_index_without_boolean_scalars(index: int, n_sessions: int) -> int:
        if _is_boolean_scalar(index):
            raise ValueError("session index must be an integer, got boolean")
        return original_validate_session_index(index, n_sessions)

    setattr(_validate_session_index_without_boolean_scalars, _PATCH_MARKER, True)
    setattr(
        _validate_session_index_without_boolean_scalars,
        "_bayescatrack_original",
        original_validate_session_index,
    )
    _growth._validate_session_index = _validate_session_index_without_boolean_scalars


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if not isinstance(value, np.ndarray):
        return False
    array = np.asarray(value, dtype=object)
    if array.shape == ():
        return isinstance(array.item(), (bool, np.bool_))
    if array.size == 1:
        return isinstance(array.reshape(-1)[0], (bool, np.bool_))
    return False


__all__ = ["install_growth_session_index_validation"]
