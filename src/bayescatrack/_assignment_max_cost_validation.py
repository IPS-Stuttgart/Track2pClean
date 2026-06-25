"""Strict validation for matching assignment cost-gate controls.

The public matching solver treats ``max_cost`` as an optional non-negative
numeric gate. Python and NumPy booleans are numeric enough for ``float(...)``,
which can silently turn ``True`` into ``1.0`` and ``False`` into ``0.0``. Reject
them explicitly so callers cannot change assignment cardinality by passing a
boolean by mistake.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_assignment_max_cost_validation_patch"
_ERROR_MESSAGE = "max_cost must be a finite non-negative value or None"


def install_assignment_max_cost_validation() -> None:
    """Install an idempotent validation wrapper around matching cost gates."""

    from . import matching as _matching  # pylint: disable=import-outside-toplevel

    original = _matching.solve_bundle_linear_assignment
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def solve_bundle_linear_assignment_with_max_cost_validation(
        bundle: Any,
        *,
        max_cost: Any = _matching.DEFAULT_ASSIGNMENT_MAX_COST,
    ) -> _matching.SessionMatchResult:
        return original(bundle, max_cost=_normalize_assignment_max_cost(max_cost))

    setattr(solve_bundle_linear_assignment_with_max_cost_validation, _PATCH_MARKER, True)
    setattr(
        solve_bundle_linear_assignment_with_max_cost_validation,
        "_bayescatrack_original",
        original,
    )
    _matching.solve_bundle_linear_assignment = (
        solve_bundle_linear_assignment_with_max_cost_validation
    )


def _normalize_assignment_max_cost(value: Any) -> float | None:
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


__all__ = ["install_assignment_max_cost_validation"]
