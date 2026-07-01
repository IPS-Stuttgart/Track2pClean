"""Validation helpers for matching assignment cost-gate controls.

The public matching solver treats ``max_cost`` as an optional non-negative
numeric gate. Python/NumPy booleans and text/binary scalar types are numeric
enough for ``float(...)`` in common edge cases, which can silently turn
``True`` into ``1.0`` or ``np.str_("1.0")`` into ``1.0``. Reject them explicitly
so callers cannot change assignment cardinality with ambiguous scalar controls.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_ERROR_MESSAGE = "max_cost must be None or a finite non-negative scalar"
_AMBIGUOUS_SCALAR_TYPES = (bool, np.bool_, str, bytes, bytearray, np.str_, np.bytes_)


def normalize_assignment_max_cost(value: Any) -> float | None:
    """Return a normalized assignment gate, rejecting ambiguous scalars."""

    if value is None:
        return None
    if isinstance(value, _AMBIGUOUS_SCALAR_TYPES):
        raise ValueError(_ERROR_MESSAGE)
    try:
        value_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if value_array.shape != ():
        raise ValueError(_ERROR_MESSAGE)

    scalar = value_array.item()
    if isinstance(scalar, _AMBIGUOUS_SCALAR_TYPES):
        raise ValueError(_ERROR_MESSAGE)
    try:
        normalized = float(scalar)
    except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(_ERROR_MESSAGE)
    return normalized


__all__ = ["normalize_assignment_max_cost"]
