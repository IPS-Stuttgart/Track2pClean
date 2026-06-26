"""Boolean-control validation for subject NPZ export options."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_EXPORT_BOOL_DEFAULTS: dict[str, bool] = {
    "include_behavior": True,
    "include_masks": False,
    "weighted": False,
    "validate_pyrecest": False,
}
_PATCH_ATTR = "_bayescatrack_subject_export_bool_validation_patch"


def install_subject_export_bool_validation(bridge_impl: Any) -> None:
    """Install idempotent validation for public subject-export boolean controls."""

    original_export = bridge_impl.export_subject_to_npz
    if getattr(original_export, _PATCH_ATTR, False):
        return

    @wraps(original_export)
    def export_subject_to_npz_with_bool_validation(*args: Any, **kwargs: Any) -> Any:
        validated_kwargs = dict(kwargs)
        for name, default in _EXPORT_BOOL_DEFAULTS.items():
            validated_kwargs[name] = _strict_export_bool(
                validated_kwargs.get(name, default),
                name=name,
            )
        return original_export(*args, **validated_kwargs)

    setattr(export_subject_to_npz_with_bool_validation, _PATCH_ATTR, True)
    setattr(
        export_subject_to_npz_with_bool_validation,
        "_bayescatrack_original",
        original_export,
    )
    bridge_impl.export_subject_to_npz = export_subject_to_npz_with_bool_validation


def _strict_export_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


__all__ = ["install_subject_export_bool_validation"]
