"""Reject duplicate session subset selectors for track scoring.

Duplicate-aware scoring treats requested session subsets as selectors, not weights.
Accepting duplicate ``session_pairs`` or duplicate ``complete_session_indices`` can
therefore double-count the same evidence and make subset diagnostics depend on
accidental repeated configuration entries.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from types import ModuleType
from typing import Any

_PATCH_ATTR = "_bayescatrack_track_subset_duplicate_validation_patch"


def install_track_subset_duplicate_validation(scores_module: ModuleType) -> None:
    """Install idempotent duplicate checks for score-track subset selectors."""

    if getattr(scores_module, _PATCH_ATTR, False):
        return

    original_normalize_session_pairs = scores_module._normalize_session_pairs
    original_normalize_complete_session_indices = (
        scores_module._normalize_complete_session_indices
    )

    def _normalize_session_pairs_without_duplicates(
        session_pairs: Any,
    ) -> tuple[tuple[int, int], ...] | None:
        normalized = original_normalize_session_pairs(session_pairs)
        if normalized is not None:
            _reject_duplicate_entries(normalized, name="session_pairs")
        return normalized

    def _normalize_complete_session_indices_without_duplicates(
        session_indices: Any,
    ) -> tuple[int, ...] | None:
        normalized = original_normalize_complete_session_indices(session_indices)
        if normalized is not None:
            _reject_duplicate_entries(normalized, name="complete_session_indices")
        return normalized

    setattr(
        _normalize_session_pairs_without_duplicates,
        _PATCH_ATTR,
        True,
    )
    setattr(
        _normalize_session_pairs_without_duplicates,
        "_bayescatrack_original",
        original_normalize_session_pairs,
    )
    setattr(
        _normalize_complete_session_indices_without_duplicates,
        _PATCH_ATTR,
        True,
    )
    setattr(
        _normalize_complete_session_indices_without_duplicates,
        "_bayescatrack_original",
        original_normalize_complete_session_indices,
    )
    scores_module._normalize_session_pairs = _normalize_session_pairs_without_duplicates
    scores_module._normalize_complete_session_indices = (
        _normalize_complete_session_indices_without_duplicates
    )
    setattr(scores_module, _PATCH_ATTR, True)


def _reject_duplicate_entries(values: Sequence[Hashable], *, name: str) -> None:
    seen: set[Hashable] = set()
    duplicates: list[Hashable] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        duplicate_summary = ", ".join(repr(value) for value in duplicates)
        raise ValueError(
            f"{name} must not contain duplicate entries; duplicate values: {duplicate_summary}"
        )


__all__ = ["install_track_subset_duplicate_validation"]
