"""Strict numeric validation for soft-overlap controls.

The soft-overlap wrappers historically used bare ``float(...)`` coercion for
runtime and preset controls.  That lets byte-like values such as ``b"0.5"``
through as valid numbers and leaks raw conversion exceptions for opaque objects
or overflowing numeric adapters.  Patch the shared helper so these public
controls fail with the same ``ValueError`` contract used by the rest of the
package validators.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_soft_overlap_numeric_validation_patch"


def install_soft_overlap_numeric_validation() -> None:
    """Install idempotent strict float validation for soft-overlap modules."""

    from . import soft_overlap_costs  # pylint: disable=import-outside-toplevel
    from .association import soft_overlap  # pylint: disable=import-outside-toplevel

    _patch_module(soft_overlap_costs)
    _patch_module(soft_overlap)


def _patch_module(module: Any) -> None:
    current = getattr(module, "_finite_float", None)
    if getattr(current, _PATCH_MARKER, False):
        return
    setattr(module, "_finite_float", _strict_finite_float)


def _strict_finite_float(
    value: Any, *, name: str, lower_bound: float, positive: bool
) -> float:
    qualifier = "positive" if positive else "non-negative"
    message = f"{name} must be a finite {qualifier} value"
    if isinstance(value, (bool, np.bool_, bytes, bytearray)):
        raise ValueError(message)
    array_value = np.asarray(value)
    if array_value.ndim > 0 or array_value.dtype.kind == "?":
        raise ValueError(message)
    scalar_value = array_value.item()
    if np.asarray(scalar_value).dtype.kind == "?":
        raise ValueError(message)
    try:
        numeric_value = float(scalar_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    violates_bound = (
        numeric_value <= lower_bound if positive else numeric_value < lower_bound
    )
    if not np.isfinite(numeric_value) or violates_bound:
        raise ValueError(message)
    return numeric_value


setattr(_strict_finite_float, _PATCH_MARKER, True)


__all__ = ["install_soft_overlap_numeric_validation"]
