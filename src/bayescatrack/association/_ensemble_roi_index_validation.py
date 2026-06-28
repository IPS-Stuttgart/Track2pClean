"""Strict validation for ensemble track-matrix ROI index scalars.

The ensemble helpers accept object-valued track matrices so CSV-derived ROI labels,
NumPy scalars, and explicit missing markers can be normalized consistently.  The
raw fallback used ``int(value)`` for otherwise unknown scalar objects, which lets
``Decimal('1.5')`` and ``Fraction(3, 2)`` truncate to ROI ``1`` instead of being
rejected as malformed labels.  This patch keeps exact integral Decimal/Fraction
inputs usable while failing fast for fractional values.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

_PATCH_ATTR = "_bayescatrack_ensemble_roi_index_validation_patch"


def install_ensemble_roi_index_validation() -> None:
    """Install an idempotent validator around ensemble ROI label parsing."""

    from . import ensemble_tracking as _ensemble_tracking  # pylint: disable=import-outside-toplevel

    original = _ensemble_tracking._roi_index_or_none  # pylint: disable=protected-access
    if getattr(original, _PATCH_ATTR, False):
        return

    def _roi_index_or_none_with_decimal_fraction_validation(value: object) -> int | None:
        scalar = _unwrap_scalar_object_array(value)
        if isinstance(scalar, (Decimal, Fraction)):
            return _coerce_exact_decimal_or_fraction_roi_index(scalar)
        return original(value)

    setattr(_roi_index_or_none_with_decimal_fraction_validation, _PATCH_ATTR, True)
    setattr(
        _roi_index_or_none_with_decimal_fraction_validation,
        "_bayescatrack_original",
        original,
    )
    _ensemble_tracking._roi_index_or_none = (  # pylint: disable=protected-access
        _roi_index_or_none_with_decimal_fraction_validation
    )


def _unwrap_scalar_object_array(value: object) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        return np.asarray(value, dtype=object).item()
    return value


def _coerce_exact_decimal_or_fraction_roi_index(value: Decimal | Fraction) -> int | None:
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError(f"track matrix contains non-integer ROI index: {value!r}")
        roi_index = int(value)
    else:
        if value.denominator != 1:
            raise ValueError(f"track matrix contains non-integer ROI index: {value!r}")
        roi_index = int(value.numerator)
    if roi_index < 0:
        return None
    return roi_index


__all__ = ["install_ensemble_roi_index_validation"]
