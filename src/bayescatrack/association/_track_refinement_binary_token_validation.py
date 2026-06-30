"""Reject binary text-like object tokens in track-refinement rows.

Track-refinement row matrices represent Suite2p ROI indices plus the configured
missing-value sentinel.  Object-dtype matrices already reject strings and bytes
before numeric normalization, but bytearray and memoryview values can still be
coerced by ``np.asarray(..., dtype=float)`` into ordinary ROI indices.  Treat
those binary text-like objects as ambiguous row tokens as well.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_refinement_binary_token_validation_patch"
_BINARY_TEXTLIKE_TYPES = (bytearray, memoryview)


def install_track_refinement_binary_token_validation() -> None:
    """Install idempotent validation for binary text-like track-row tokens."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original_contains_ambiguous_tokens = (
        module._contains_ambiguous_track_row_tokens  # pylint: disable=protected-access
    )
    if getattr(original_contains_ambiguous_tokens, _PATCH_MARKER, False):
        return

    @wraps(original_contains_ambiguous_tokens)
    def contains_ambiguous_track_row_tokens_with_binary_validation(
        rows: np.ndarray,
    ) -> bool:
        if original_contains_ambiguous_tokens(rows):
            return True
        if rows.dtype != object:
            return False
        return any(isinstance(value, _BINARY_TEXTLIKE_TYPES) for value in rows.ravel())

    setattr(
        contains_ambiguous_track_row_tokens_with_binary_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        contains_ambiguous_track_row_tokens_with_binary_validation,
        "_bayescatrack_original",
        original_contains_ambiguous_tokens,
    )
    module._contains_ambiguous_track_row_tokens = (  # pylint: disable=protected-access
        contains_ambiguous_track_row_tokens_with_binary_validation
    )


__all__ = ["install_track_refinement_binary_token_validation"]
