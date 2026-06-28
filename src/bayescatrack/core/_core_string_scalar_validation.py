"""Reject ambiguous string-like values for core numeric scalar controls.

The core scalar validator accepts numeric Python and NumPy scalars, but string-like
values are configuration artefacts rather than numeric controls.  Python's
``float(...)`` accepts values such as ``"1.0"`` and ``b"1.0"``, so reject them
before they can silently alter pairwise costs or exported state covariances.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_core_string_scalar_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_STRINGLIKE_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_core_string_scalar_validation(core_scalar_validation_module: Any) -> None:
    """Install an idempotent guard around core finite-scalar validation."""

    original = (
        core_scalar_validation_module._validate_finite_scalar
    )  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    def _validate_finite_scalar_without_strings(
        name: str,
        raw_value: Any,
        *,
        strictly_positive: bool,
    ) -> float:
        if _is_stringlike_scalar(raw_value):
            raise ValueError(f"{name} must be {_scalar_requirement(strictly_positive)}")
        return original(name, raw_value, strictly_positive=strictly_positive)

    setattr(_validate_finite_scalar_without_strings, _PATCH_MARKER, True)
    setattr(_validate_finite_scalar_without_strings, _ORIGINAL_ATTR, original)
    core_scalar_validation_module._validate_finite_scalar = (  # pylint: disable=protected-access
        _validate_finite_scalar_without_strings
    )


def _is_stringlike_scalar(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return False
        value = value.item()
    return isinstance(value, _STRINGLIKE_SCALAR_TYPES)


def _scalar_requirement(strictly_positive: bool) -> str:
    if strictly_positive:
        return "a finite positive value"
    return "a finite non-negative value"


__all__ = ["install_core_string_scalar_validation"]
