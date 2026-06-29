"""Strict numeric validation for soft-overlap and registered-mask controls.

The soft-overlap wrappers historically used bare ``float(...)`` coercion for
runtime and preset controls.  That lets byte-like values such as ``b"0.5"``
through as valid numbers and leaks raw conversion exceptions for opaque objects
or overflowing numeric adapters.  Patch the shared helpers so these public
controls fail with the same ``ValueError`` contract used by the rest of the
package validators.  The registered-mask helpers use the same large-cost control
in global registered-pairwise pipelines, so they are patched here as well.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_soft_overlap_numeric_validation_patch"
_COERCION_EXCEPTIONS = (TypeError, ValueError, OverflowError, Exception)


def install_soft_overlap_numeric_validation() -> None:
    """Install idempotent strict scalar validation for soft-overlap modules."""

    from . import soft_overlap_costs  # pylint: disable=import-outside-toplevel
    from .association import registered_masks  # pylint: disable=import-outside-toplevel
    from .association import soft_overlap  # pylint: disable=import-outside-toplevel

    _patch_module(soft_overlap_costs)
    _patch_module(soft_overlap)
    _patch_registered_mask_module(registered_masks)


def _patch_module(module: Any) -> None:
    current_float = getattr(module, "_finite_float", None)
    if not getattr(current_float, _PATCH_MARKER, False):
        setattr(module, "_finite_float", _strict_finite_float)

    current_int = getattr(module, "_nonnegative_int", None)
    if current_int is not None and not getattr(current_int, _PATCH_MARKER, False):
        setattr(module, "_nonnegative_int", _strict_nonnegative_int)


def _patch_registered_mask_module(module: Any) -> None:
    current_float = getattr(module, "_finite_positive_float", None)
    if current_float is not None and not getattr(current_float, _PATCH_MARKER, False):
        setattr(module, "_finite_positive_float", _strict_finite_positive_float)


def _strict_numeric_scalar(value: Any, *, message: str) -> Any:
    if isinstance(value, (bool, np.bool_, bytes, bytearray)):
        raise ValueError(message)
    try:
        array_value = np.asarray(value)
    except _COERCION_EXCEPTIONS as exc:
        raise ValueError(message) from exc
    if array_value.ndim > 0 or array_value.dtype.kind == "?":
        raise ValueError(message)
    try:
        scalar_value = array_value.item()
    except _COERCION_EXCEPTIONS as exc:
        raise ValueError(message) from exc
    if isinstance(scalar_value, (bool, np.bool_, bytes, bytearray)):
        raise ValueError(message)
    try:
        scalar_kind = np.asarray(scalar_value).dtype.kind
    except _COERCION_EXCEPTIONS as exc:
        raise ValueError(message) from exc
    if scalar_kind == "?":
        raise ValueError(message)
    return scalar_value


def _strict_finite_float(
    value: Any, *, name: str, lower_bound: float, positive: bool
) -> float:
    qualifier = "positive" if positive else "non-negative"
    message = f"{name} must be a finite {qualifier} value"
    scalar_value = _strict_numeric_scalar(value, message=message)
    try:
        numeric_value = float(scalar_value)
    except _COERCION_EXCEPTIONS as exc:
        raise ValueError(message) from exc
    violates_bound = (
        numeric_value <= lower_bound if positive else numeric_value < lower_bound
    )
    if not np.isfinite(numeric_value) or violates_bound:
        raise ValueError(message)
    return numeric_value


def _strict_finite_positive_float(value: Any, *, name: str) -> float:
    return _strict_finite_float(value, name=name, lower_bound=0.0, positive=True)


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    message = f"{name} must be an integer"
    scalar_value = _strict_numeric_scalar(value, message=message)
    numeric_candidate: Any
    if isinstance(scalar_value, str):
        numeric_candidate = scalar_value.strip()
        if not numeric_candidate:
            raise ValueError(message)
    elif isinstance(scalar_value, (float, np.floating)):
        numeric_candidate = scalar_value
    else:
        try:
            return _reject_negative_int(operator.index(scalar_value), name=name)
        except _COERCION_EXCEPTIONS:
            try:
                numeric_candidate = float(scalar_value)
            except _COERCION_EXCEPTIONS as exc:
                raise ValueError(message) from exc
    try:
        numeric_value = float(numeric_candidate)
    except _COERCION_EXCEPTIONS as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(message)
    return _reject_negative_int(int(numeric_value), name=name)


def _reject_negative_int(integer_value: int, *, name: str) -> int:
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(integer_value)


setattr(_strict_finite_float, _PATCH_MARKER, True)
setattr(_strict_finite_positive_float, _PATCH_MARKER, True)
setattr(_strict_nonnegative_int, _PATCH_MARKER, True)

__all__ = ["install_soft_overlap_numeric_validation"]
