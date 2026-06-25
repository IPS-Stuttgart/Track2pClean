"""Strict validation for TrackTable session-name uniqueness.

Track-table scoring aligns prediction columns to ground-truth columns by session
name.  Duplicate session names are ambiguous: set-based equality can hide them,
and tuple.index(...) selects only the first duplicate column.  The validation hook
below makes that ambiguity explicit before scoring or column reordering can use the
wrong session column silently.
"""

from __future__ import annotations

from typing import Any, Sequence

_PATCH_MARKER = "_bayescatrack_track_table_session_name_validation_patch"


def install_track_table_session_name_validation() -> None:
    """Install idempotent validation around ground-truth TrackTable helpers."""

    from . import ground_truth_eval as _ground_truth_eval  # pylint: disable=import-outside-toplevel

    track_table = _ground_truth_eval.TrackTable
    if getattr(track_table, _PATCH_MARKER, False):
        return

    original_init = track_table.__init__
    original_aligned_to = track_table.aligned_to

    def __init__(self: Any, session_names: Sequence[str], tracks: Any) -> None:
        normalized_session_names = _normalize_unique_session_names(
            session_names,
            field_name="session_names",
        )
        original_init(self, normalized_session_names, tracks)

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

    setattr(track_table, "__init__", __init__)
    setattr(track_table, "aligned_to", aligned_to)
    setattr(track_table, _PATCH_MARKER, True)


def _normalize_unique_session_names(
    session_names: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized_session_names = tuple(str(name) for name in session_names)
    seen: set[str] = set()
    duplicates: list[str] = []
    for session_name in normalized_session_names:
        if session_name in seen and session_name not in duplicates:
            duplicates.append(session_name)
        seen.add(session_name)
    if duplicates:
        duplicate_summary = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            f"{field_name} must contain unique session names; duplicate values: {duplicate_summary}"
        )
    return normalized_session_names


__all__ = ["install_track_table_session_name_validation"]
