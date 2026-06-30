"""Strict session-index validation for Track2p-teacher prior edges."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_teacher_prior_index_validation_patch"


def install_teacher_prior_index_validation() -> None:
    """Install idempotent validation for teacher-prior session edge indices."""

    from . import teacher_priors as module  # pylint: disable=import-outside-toplevel

    original = module._normalize_session_index  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _normalize_session_index_with_validation(
        value: Any,
        *,
        context: str,
        session_count: int,
    ) -> int:
        return _normalize_session_index(
            value,
            context=context,
            session_count=session_count,
        )

    setattr(_normalize_session_index_with_validation, _PATCH_MARKER, True)
    setattr(_normalize_session_index_with_validation, "_bayescatrack_original", original)
    module._normalize_session_index = _normalize_session_index_with_validation  # pylint: disable=protected-access


def _normalize_session_index(value: Any, *, context: str, session_count: int) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{context} must be an integer session index")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be an integer session index")

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{context} must be an integer session index")
        index = int(numeric)
    else:
        try:
            index = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context} must be an integer session index") from exc

    index = int(index)
    if index < 0 or index >= session_count:
        raise ValueError(f"{context} {index} out of bounds for {session_count} sessions")
    return index


__all__ = ["install_teacher_prior_index_validation"]
