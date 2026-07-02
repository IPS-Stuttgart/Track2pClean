"""Reject bytes-like multi-hypothesis edge controls.

The multi-hypothesis helpers treat edge-like inputs as short iterables.  Without
an explicit bytes-like guard, ``bytearray`` and ``memoryview`` inputs can be
unpacked as integer byte values and accidentally become valid session/ROI edge
identifiers.  Reject those ambiguous buffers before tuple unpacking.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from . import multi_hypothesis as _multi_hypothesis

_PATCH_MARKER = "_bayescatrack_multi_hypothesis_bytes_like_validation_patch"
_BYTES_LIKE_SEQUENCE_TYPES = (bytearray, memoryview)


def install_multi_hypothesis_bytes_like_validation() -> None:
    """Install idempotent bytes-like edge validation patches."""

    _patch_session_edge_validator()
    _patch_edge_candidate_validator()
    _patch_consensus_edge_validator()


def _patch_session_edge_validator() -> None:
    original = (
        _multi_hypothesis._validated_session_edge
    )  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated_session_edge(edge: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(edge, _BYTES_LIKE_SEQUENCE_TYPES):
            raise ValueError(
                f"{kwargs.get('name', 'edge')} must be a two-item session edge"
            )
        return original(edge, *args, **kwargs)

    setattr(validated_session_edge, _PATCH_MARKER, True)
    setattr(validated_session_edge, "_bayescatrack_original", original)
    _multi_hypothesis._validated_session_edge = (
        validated_session_edge  # pylint: disable=protected-access
    )


def _patch_edge_candidate_validator() -> None:
    original = (
        _multi_hypothesis._validated_edge_candidate
    )  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated_edge_candidate(candidate: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(candidate, _BYTES_LIKE_SEQUENCE_TYPES):
            raise ValueError(
                f"{kwargs.get('name', 'candidate')} must be a three-item edge candidate"
            )
        return original(candidate, *args, **kwargs)

    setattr(validated_edge_candidate, _PATCH_MARKER, True)
    setattr(validated_edge_candidate, "_bayescatrack_original", original)
    _multi_hypothesis._validated_edge_candidate = (
        validated_edge_candidate  # pylint: disable=protected-access
    )


def _patch_consensus_edge_validator() -> None:
    original = (
        _multi_hypothesis._normalize_consensus_edge
    )  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def normalize_consensus_edge(edge: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(edge, _BYTES_LIKE_SEQUENCE_TYPES):
            raise ValueError(
                f"{kwargs.get('name', 'edge')} must be a four-item consensus edge"
            )
        return original(edge, *args, **kwargs)

    setattr(normalize_consensus_edge, _PATCH_MARKER, True)
    setattr(normalize_consensus_edge, "_bayescatrack_original", original)
    _multi_hypothesis._normalize_consensus_edge = (
        normalize_consensus_edge  # pylint: disable=protected-access
    )


__all__ = ["install_multi_hypothesis_bytes_like_validation"]
