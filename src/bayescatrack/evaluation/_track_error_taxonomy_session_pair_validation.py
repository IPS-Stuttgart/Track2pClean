"""Session-pair validation hook for track-error taxonomy."""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType

from .complete_track_scores import _coerce_integer_like_index

_PATCH_ATTR = "_bayescatrack_taxonomy_session_pair_validation_patch"


def install_track_error_taxonomy_session_pair_validation(
    taxonomy_module: ModuleType,
) -> None:
    """Reject malformed taxonomy ``session_pairs`` before link counting."""

    original_session_pairs = (
        taxonomy_module._session_pairs
    )  # pylint: disable=protected-access
    if getattr(original_session_pairs, _PATCH_ATTR, False):
        return

    def _session_pairs_with_strict_indices(
        n_sessions: int,
        session_pairs: Iterable[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        if session_pairs is None:
            return original_session_pairs(n_sessions, session_pairs)
        normalized_pairs = tuple(
            (_coerce_session_index(source), _coerce_session_index(target))
            for source, target in session_pairs
        )
        return original_session_pairs(n_sessions, normalized_pairs)

    setattr(_session_pairs_with_strict_indices, _PATCH_ATTR, True)
    setattr(
        _session_pairs_with_strict_indices,
        "_bayescatrack_original",
        original_session_pairs,
    )
    taxonomy_module._session_pairs = (
        _session_pairs_with_strict_indices  # pylint: disable=protected-access
    )


def _coerce_session_index(value: object) -> int:
    return _coerce_integer_like_index(
        value,
        context="session_pairs",
        index_kind="session",
        requirement="Session indices must be integer-like",
    )
