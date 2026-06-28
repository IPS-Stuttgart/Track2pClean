"""Validate reference session-index selector containers."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

_PATCH_ATTR = "_bayescatrack_reference_session_selector_validation_patch"
_ERROR_MESSAGE = "session_indices must be an iterable of integer session indices"
_BYTES_LIKE_SELECTOR_TYPES = (bytearray, memoryview)


def install_reference_session_selector_validation(
    reference_module: ModuleType | None = None,
) -> None:
    """Install idempotent validation for malformed ``session_indices`` containers."""

    if reference_module is None:
        from . import (  # pylint: disable=import-outside-toplevel,reimported
            reference as reference_module,
        )

    original_normalize_session_indices = (
        reference_module._normalize_session_indices
    )  # pylint: disable=protected-access
    if getattr(original_normalize_session_indices, _PATCH_ATTR, False):
        return

    @wraps(original_normalize_session_indices)
    def _normalize_session_indices_with_selector_validation(
        session_indices: Any,
        n_sessions: int,
    ) -> tuple[int, ...]:
        if isinstance(session_indices, _BYTES_LIKE_SELECTOR_TYPES):
            raise ValueError(_ERROR_MESSAGE)
        return original_normalize_session_indices(session_indices, n_sessions)

    setattr(_normalize_session_indices_with_selector_validation, _PATCH_ATTR, True)
    setattr(
        _normalize_session_indices_with_selector_validation,
        "_bayescatrack_original",
        original_normalize_session_indices,
    )
    reference_module._normalize_session_indices = (  # pylint: disable=protected-access
        _normalize_session_indices_with_selector_validation
    )


__all__ = ["install_reference_session_selector_validation"]