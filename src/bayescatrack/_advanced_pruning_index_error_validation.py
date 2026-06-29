"""Normalize malformed ``__index__`` failures for advanced pruning controls."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_advanced_pruning_index_error_validation_patch"
_ERROR_MESSAGE = "{name} must be a positive integer or None"


def install_advanced_pruning_index_error_validation() -> None:
    """Install an idempotent adapter around advanced top-k integer parsing."""

    from . import advanced_roi_components as advanced  # pylint: disable=import-outside-toplevel

    original = advanced._normalize_optional_positive_int  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def normalize_optional_positive_int(value: Any, *, name: str) -> int | None:
        try:
            return original(value, name=name)
        except (ValueError, OverflowError) as exc:
            raise ValueError(_ERROR_MESSAGE.format(name=name)) from exc

    setattr(normalize_optional_positive_int, _PATCH_MARKER, True)
    setattr(normalize_optional_positive_int, "_bayescatrack_original", original)
    advanced._normalize_optional_positive_int = (  # pylint: disable=protected-access
        normalize_optional_positive_int
    )


__all__ = ["install_advanced_pruning_index_error_validation"]
