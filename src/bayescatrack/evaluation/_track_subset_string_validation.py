"""Reject ambiguous or malformed session subset selectors for aggregate track scoring."""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import Any

_PATCH_ATTR = "_bayescatrack_track_subset_string_validation_patch"
_STRING_LIKE_TYPES = (str, bytes, bytearray)


def install_track_subset_string_validation(scores_module: ModuleType) -> None:
    """Install idempotent guards for ambiguous subset selectors."""

    if getattr(scores_module, _PATCH_ATTR, False):
        return

    original_normalize_session_pairs = scores_module._normalize_session_pairs
    original_normalize_complete_session_indices = (
        scores_module._normalize_complete_session_indices
    )

    def _normalize_session_pairs_without_strings(
        session_pairs: Any,
    ) -> tuple[tuple[int, int], ...] | None:
        _reject_string_like_selector(
            session_pairs,
            name="session_pairs",
            expected="an iterable of session-index pairs",
        )
        if session_pairs is None:
            return original_normalize_session_pairs(session_pairs)

        normalized_pairs = tuple(
            _normalize_session_pair(pair)
            for pair in _materialize_selector(
                session_pairs,
                name="session_pairs",
                expected="an iterable of session-index pairs",
            )
        )
        return original_normalize_session_pairs(normalized_pairs)

    def _normalize_complete_session_indices_without_strings(
        session_indices: Any,
    ) -> tuple[int, ...] | None:
        _reject_string_like_selector(
            session_indices,
            name="complete_session_indices",
            expected="a sequence of session indices",
        )
        if session_indices is None:
            return original_normalize_complete_session_indices(session_indices)
        return original_normalize_complete_session_indices(
            _materialize_selector(
                session_indices,
                name="complete_session_indices",
                expected="a sequence of session indices",
            )
        )

    setattr(_normalize_session_pairs_without_strings, _PATCH_ATTR, True)
    setattr(
        _normalize_session_pairs_without_strings,
        "_bayescatrack_original",
        original_normalize_session_pairs,
    )
    setattr(_normalize_complete_session_indices_without_strings, _PATCH_ATTR, True)
    setattr(
        _normalize_complete_session_indices_without_strings,
        "_bayescatrack_original",
        original_normalize_complete_session_indices,
    )
    scores_module._normalize_session_pairs = _normalize_session_pairs_without_strings
    scores_module._normalize_complete_session_indices = (
        _normalize_complete_session_indices_without_strings
    )
    setattr(scores_module, _PATCH_ATTR, True)


def _normalize_session_pair(pair: Any) -> tuple[Any, Any]:
    _reject_string_like_selector(
        pair,
        name="session_pairs entries",
        expected="two-item session-index iterables",
    )
    try:
        source, target = pair
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "session_pairs entries must be two-item session-index iterables"
        ) from exc
    return source, target


def _materialize_selector(
    value: Iterable[Any], *, name: str, expected: str
) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be {expected}") from exc


def _reject_string_like_selector(value: Any, *, name: str, expected: str) -> None:
    if value is None:
        return
    if isinstance(value, _STRING_LIKE_TYPES):
        raise ValueError(f"{name} must be {expected}, not a bare string-like value")


__all__ = ["install_track_subset_string_validation"]
