from __future__ import annotations

from typing import Any

_PATCH_MARKER = "_bayescatrack_session_name_entry_validation_patch"
_STRING_LIKE_SESSION_NAMES = (str, bytes, bytearray)
_ERROR_MESSAGE = "session_names must contain non-empty string session names"


def install_session_name_entry_validation() -> None:
    """Install stricter checks for individual session-name entries."""

    from . import _track_table_session_name_validation as module

    original = module._normalize_unique_session_names  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    def _normalize_unique_session_names_with_entry_validation(
        session_names: Any,
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        if not isinstance(session_names, _STRING_LIKE_SESSION_NAMES):
            _validate_raw_session_name_entries(session_names)
        return original(session_names, field_name=field_name)

    setattr(_normalize_unique_session_names_with_entry_validation, _PATCH_MARKER, True)
    setattr(
        _normalize_unique_session_names_with_entry_validation,
        "_bayescatrack_original",
        original,
    )
    module._normalize_unique_session_names = (  # pylint: disable=protected-access
        _normalize_unique_session_names_with_entry_validation
    )


def _validate_raw_session_name_entries(session_names: Any) -> None:
    try:
        raw_names = tuple(session_names)
    except TypeError:
        return
    for value in raw_names:
        if not isinstance(value, str) or value == "":
            raise ValueError(_ERROR_MESSAGE)


__all__ = ["install_session_name_entry_validation"]
