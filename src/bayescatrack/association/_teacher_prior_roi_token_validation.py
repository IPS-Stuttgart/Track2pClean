"""Teacher-prior ROI token validation for binary buffer inputs.

Track2p teacher-prior matrices may contain text/bytes ROI identifiers or missing
sentinels.  Python accepts ``float(memoryview(b"1"))`` and
``float(bytearray(b"1"))``, so those binary buffer objects can otherwise be
silently interpreted as real ROI indices by the fallback parser.  Treat them as
malformed/missing ROI cells instead, matching the existing missing-cell behavior
for unparseable teacher entries.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_teacher_prior_roi_token_validation_patch"
_BINARY_BUFFER_TYPES = (bytearray, memoryview)


def install_teacher_prior_roi_token_validation() -> None:
    """Install idempotent validation for teacher-prior ROI index cells."""

    from . import teacher_priors as module  # pylint: disable=import-outside-toplevel

    original = module._parse_roi_index  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _parse_roi_index_with_binary_buffer_guard(value: Any) -> int | None:
        candidate = value
        if isinstance(candidate, np.ndarray):
            if candidate.shape == ():
                candidate = candidate.item()
            else:
                return original(value)
        if isinstance(candidate, _BINARY_BUFFER_TYPES):
            return None
        return original(value)

    setattr(_parse_roi_index_with_binary_buffer_guard, _PATCH_MARKER, True)
    setattr(
        _parse_roi_index_with_binary_buffer_guard, "_bayescatrack_original", original
    )
    module._parse_roi_index = (
        _parse_roi_index_with_binary_buffer_guard  # pylint: disable=protected-access
    )


__all__ = ["install_teacher_prior_roi_token_validation"]
