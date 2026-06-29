"""Strict validation for Track2p subject-loader input-format controls.

Subject loading accepts an ``input_format`` mode selector.  The lower-level loader
checks membership in the allowed mode set directly, so unhashable malformed values
such as lists can raise a raw ``TypeError`` before the public validation error is
reported.  This wrapper normalizes valid string-like values and rejects everything
else at the subject-loader boundary.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_ALLOWED_INPUT_FORMATS = frozenset({"auto", "suite2p", "npy"})
_ERROR_MESSAGE = "input_format must be 'auto', 'suite2p', or 'npy'"
_PATCH_MARKER = "_bayescatrack_subject_input_format_validation_patch"


def install_subject_input_format_validation(bridge_impl: ModuleType) -> None:
    """Install an idempotent validator around subject loading."""

    _patch_input_format_keyword_function(bridge_impl, "load_track2p_subject")


def install_subject_like_input_format_validation(bridge_module: ModuleType) -> None:
    """Install input-format validation around public subject-like entrypoints."""

    for function_name in (
        "load_track2p_subject",
        "export_subject_to_npz",
        "summarize_subject",
    ):
        if hasattr(bridge_module, function_name):
            _patch_input_format_keyword_function(bridge_module, function_name)


def _patch_input_format_keyword_function(module: ModuleType, function_name: str) -> None:
    original = getattr(module, function_name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def subject_like_with_input_format_validation(
        subject_dir: str | Path,
        *args: Any,
        input_format: Any = "auto",
        **kwargs: Any,
    ) -> Any:
        return original(
            subject_dir,
            *args,
            input_format=_normalize_input_format(input_format),
            **kwargs,
        )

    setattr(subject_like_with_input_format_validation, _PATCH_MARKER, True)
    if getattr(original, "_bayescatrack_auto_fallback_patch", False):
        setattr(
            subject_like_with_input_format_validation,
            "_bayescatrack_auto_fallback_patch",
            True,
        )
    setattr(
        subject_like_with_input_format_validation,
        "_bayescatrack_original",
        original,
    )
    setattr(module, function_name, subject_like_with_input_format_validation)


def _normalize_input_format(value: Any) -> str:
    if isinstance(value, np.ndarray):
        raise ValueError(_ERROR_MESSAGE)
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str) or value not in _ALLOWED_INPUT_FORMATS:
        raise ValueError(_ERROR_MESSAGE)
    return value


__all__ = [
    "install_subject_input_format_validation",
    "install_subject_like_input_format_validation",
]
