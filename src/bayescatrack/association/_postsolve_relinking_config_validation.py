"""Strict scalar-control validation for post-solve relinking config.

Post-solve relinking config controls are usually numeric Python scalars.  The
base helper used ``float(...)`` after NumPy object-scalar normalization, which
also accepts text/binary tokens such as ``"1.0"`` or ``b"1.0"``.  Reject those
values explicitly so malformed benchmark manifests do not silently change the
relinking policy.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_postsolve_relinking_config_validation"
_TEXT_OR_BINARY_TYPES = (str, bytes, bytearray, memoryview)


def install_postsolve_relinking_config_validation() -> None:
    """Install idempotent validation for post-solve relinking scalar controls."""

    from . import postsolve_relinking as postsolve_relinking_module

    original = postsolve_relinking_module._finite_nonnegative_float
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _finite_nonnegative_float_with_text_guard(value: Any, name: str) -> float:
        if _is_text_or_binary_scalar(value):
            raise ValueError(f"{name} must be finite and non-negative")
        return original(value, name)

    setattr(_finite_nonnegative_float_with_text_guard, _PATCH_MARKER, True)
    setattr(_finite_nonnegative_float_with_text_guard, "_bayescatrack_original", original)
    postsolve_relinking_module._finite_nonnegative_float = (  # pylint: disable=protected-access
        _finite_nonnegative_float_with_text_guard
    )


def _is_text_or_binary_scalar(value: Any) -> bool:
    if isinstance(value, _TEXT_OR_BINARY_TYPES):
        return True
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return False
        try:
            value = value.item()
        except ValueError:
            return False
        return isinstance(value, _TEXT_OR_BINARY_TYPES)
    return False


__all__ = ["install_postsolve_relinking_config_validation"]
