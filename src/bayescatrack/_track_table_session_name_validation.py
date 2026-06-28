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
_GROUND_TRUTH_LOADER_PATCH_MARKER = (
    "_bayescatrack_ground_truth_loader_session_name_validation_patch"
)
_STRING_LIKE_SESSION_NAMES = (str, bytes, bytearray)


def install_track_table_session_name_validation() -> None:
    """Install idempotent validation around session-name-bearing helpers."""

    from . import ground_truth_eval  # pylint: disable=import-outside-toplevel
    from . import reference  # pylint: disable=import-outside-toplevel

    _install_track_table_session_name_validation(ground_truth_eval.TrackTable)
    _install_ground_truth_loader_session_name_validation(ground_truth_eval)
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


def _install_ground_truth_loader_session_name_validation(
    ground_truth_eval: Any,
) -> None:
    _patch_optional_session_names_argument(ground_truth_eval, "load_track_table_csv")
    _patch_optional_session_names_argument(
        ground_truth_eval, "load_track2p_ground_truth_csv"
    )
    _patch_required_session_names_argument(
        ground_truth_eval, "tracks_from_consecutive_matches"
    )


def _patch_optional_session_names_argument(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if _function_chain_has_patch(original, _GROUND_TRUTH_LOADER_PATCH_MARKER):
        return

    @wraps(original)
    def _with_validated_optional_session_names(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("session_names") is not None:
            normalized_kwargs = dict(kwargs)
            normalized_kwargs["session_names"] = _normalize_unique_session_names(
                kwargs["session_names"],
                field_name="session_names",
            )
            kwargs = normalized_kwargs
        return original(*args, **kwargs)

    setattr(
        _with_validated_optional_session_names, _GROUND_TRUTH_LOADER_PATCH_MARKER, True
    )
    setattr(_with_validated_optional_session_names, "_bayescatrack_original", original)
    setattr(module, function_name, _with_validated_optional_session_names)


def _patch_required_session_names_argument(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if _function_chain_has_patch(original, _GROUND_TRUTH_LOADER_PATCH_MARKER):
        return

    @wraps(original)
    def _with_validated_required_session_names(
        session_names: Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_session_names = _normalize_unique_session_names(
            session_names,
            field_name="session_names",
        )
        return original(normalized_session_names, *args, **kwargs)

    setattr(
        _with_validated_required_session_names, _GROUND_TRUTH_LOADER_PATCH_MARKER, True
    )
    setattr(_with_validated_required_session_names, "_bayescatrack_original", original)
    setattr(module, function_name, _with_validated_required_session_names)


def _function_chain_has_patch(function: Any, marker: str) -> bool:
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, marker, False):
            return True
        seen.add(current_id)
        current = getattr(current, "_bayescatrack_original", None)
    return False


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
    if isinstance(session_names, _STRING_LIKE_SESSION_NAMES):
        raise ValueError(
            f"{field_name} must be a sequence of session-name values, not a bare string-like value"
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
