"""Strict validation for session-name uniqueness in track/reference tables.

Track-table scoring aligns prediction columns to ground-truth columns by session
name.  Duplicate session names are ambiguous: set-based equality can hide them,
and tuple.index(...) selects only the first duplicate column.  Track2pReference
objects expose the same session-name contract, so references receive the same
uniqueness guard.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Sequence

_TRACK_TABLE_PATCH_MARKER = "_bayescatrack_track_table_session_name_validation_patch"
_REFERENCE_PATCH_MARKER = "_bayescatrack_reference_session_name_validation_patch"


def install_track_table_session_name_validation() -> None:
    """Install idempotent validation around session-name-bearing helpers."""

    from . import ground_truth_eval  # pylint: disable=import-outside-toplevel
    from . import reference  # pylint: disable=import-outside-toplevel

    _install_track_table_session_name_validation(ground_truth_eval.TrackTable)
    _install_reference_session_name_validation(reference.Track2pReference)


def _install_track_table_session_name_validation(track_table: Any) -> None:
    if getattr(track_table, _TRACK_TABLE_PATCH_MARKER, False):
        return

    original_post_init = track_table.__post_init__
    original_aligned_to = track_table.aligned_to

    @wraps(original_post_init)
    def __post_init__(self: Any) -> None:
        normalized_session_names = _normalize_unique_session_names(
            self.session_names,
            field_name="session_names",
        )
        object.__setattr__(self, "session_names", normalized_session_names)
        original_post_init(self)

    def aligned_to(self: Any, session_names: Sequence[str]) -> Any:
        _normalize_unique_session_names(
            self.session_names,
            field_name="existing session_names",
        )
        normalized_session_names = _normalize_unique_session_names(
            session_names,
            field_name="target session_names",
        )
        return original_aligned_to(self, normalized_session_names)

    setattr(track_table, "__post_init__", __post_init__)
    setattr(track_table, "aligned_to", aligned_to)
    setattr(track_table, _TRACK_TABLE_PATCH_MARKER, True)


def _install_reference_session_name_validation(reference_cls: Any) -> None:
    original_post_init = reference_cls.__post_init__
    if getattr(original_post_init, _REFERENCE_PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def __post_init__(self: Any) -> None:
        normalized_session_names = _normalize_unique_session_names(
            self.session_names,
            field_name="session_names",
        )
        object.__setattr__(self, "session_names", normalized_session_names)
        original_post_init(self)

    setattr(__post_init__, _REFERENCE_PATCH_MARKER, True)
    setattr(__post_init__, "_bayescatrack_original", original_post_init)
    reference_cls.__post_init__ = __post_init__


def _normalize_unique_session_names(
    session_names: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(session_names, (str, bytes)):
        raise ValueError(
            f"{field_name} must be a sequence of session-name values, not a bare string"
        )
    try:
        normalized_session_names = tuple(str(name) for name in session_names)
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must be a sequence of session-name values"
        ) from exc
    seen: set[str] = set()
    duplicates: list[str] = []
    for session_name in normalized_session_names:
        if session_name in seen and session_name not in duplicates:
            duplicates.append(session_name)
        seen.add(session_name)
    if duplicates:
        duplicate_summary = ", ".join(repr(name) for name in duplicates)
        message = (
            f"{field_name} must contain unique session names; "
            f"duplicate values: {duplicate_summary}"
        )
        raise ValueError(message)
    return normalized_session_names


__all__ = ["install_track_table_session_name_validation"]
