"""Strict integer-error normalization for multi-hypothesis consensus inputs.

``consensus_edges`` accepts dense track matrices and explicit edge sets at an
export boundary. The underlying consensus parser previously used integer
conversion directly, so custom integer-like objects could leak ``__index__``
``ValueError``/``OverflowError``/``ArithmeticError`` exceptions instead of
raising the module's documented ``ValueError`` for malformed integer entries.
"""

from __future__ import annotations

from typing import Any

from . import multi_hypothesis as _multi_hypothesis
from ._numeric_validation import integer as _integer

_PATCH_MARKER = "_bayescatrack_consensus_integer_validation_patch"
_INTEGER_ENTRY_ERROR = "track matrices or edge sets must contain integer entries"


def install_consensus_integer_validation() -> None:
    """Install an idempotent consensus-entry integer normalizer."""

    original = _multi_hypothesis._normalize_consensus_integer_entry
    if getattr(original, _PATCH_MARKER, False):
        return

    def _normalize_consensus_integer_entry(value: Any) -> int:
        try:
            return _integer(value, name="track matrices or edge sets")
        except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
            raise ValueError(_INTEGER_ENTRY_ERROR) from exc

    setattr(_normalize_consensus_integer_entry, _PATCH_MARKER, True)
    setattr(_normalize_consensus_integer_entry, "_bayescatrack_original", original)
    _multi_hypothesis._normalize_consensus_integer_entry = (  # type: ignore[assignment]
        _normalize_consensus_integer_entry
    )


__all__ = ["install_consensus_integer_validation"]
