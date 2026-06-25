"""Validation helpers for matching assignment cost-gate controls.

The public matching solver treats ``max_cost`` as an optional non-negative
numeric gate. Python and NumPy booleans are numeric enough for ``float(...)``,
which can silently turn ``True`` into ``1.0`` and ``False`` into ``0.0``. Reject
them explicitly so callers cannot change assignment cardinality by passing a
boolean by mistake.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_ERROR_MESSAGE = "max_cost must be a finite non-negative value or None"


def normalize_assignment_max_cost(value: Any) -> float | None:
    """Return a normalized assignment gate, rejecting ambiguous booleans."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(_ERROR_MESSAGE)
    return normalized


__all__ = ["normalize_assignment_max_cost"]
