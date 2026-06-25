"""Strict validation for subject-export boolean options.

``export_subject_to_npz`` uses several option flags to choose the exported NPZ
payload and whether optional validation code is executed.  Passing strings such
as ``"false"`` is easy when options come from configuration files, but Python
truthiness would treat those strings as enabled flags.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_subject_export_option_validation_patch"
_BOOL_OPTION_DEFAULTS: dict[str, bool] = {
    "include_behavior": True,
    "include_masks": False,
    "weighted": False,
    "validate_pyrecest": False,
}


def install_subject_export_option_validation(bridge_impl: Any) -> None:
    """Install idempotent validation around ``export_subject_to_npz`` flags."""

    original_export = bridge_impl.export_subject_to_npz
    if getattr(original_export, _PATCH_ATTR, False):
        return

    @wraps(original_export)
    def export_subject_to_npz_with_option_validation(
        subject_dir: str | Path,
        output_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if args:
            return original_export(subject_dir, output_path, *args, **kwargs)
        normalized_kwargs = dict(kwargs)
        for name, default in _BOOL_OPTION_DEFAULTS.items():
            normalized_kwargs[name] = _normalize_boolean_option(
                normalized_kwargs.get(name, default),
                name=name,
            )
        return original_export(subject_dir, output_path, **normalized_kwargs)

    setattr(export_subject_to_npz_with_option_validation, _PATCH_ATTR, True)
    setattr(
        export_subject_to_npz_with_option_validation,
        "_bayescatrack_original",
        original_export,
    )
    bridge_impl.export_subject_to_npz = export_subject_to_npz_with_option_validation


def _normalize_boolean_option(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


__all__ = ["install_subject_export_option_validation"]
