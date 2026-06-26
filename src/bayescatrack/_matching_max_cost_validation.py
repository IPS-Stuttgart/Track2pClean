"""Strict validation for linear-assignment max-cost gates.

The assignment gate is a numeric threshold, not a truthiness flag.  Without an
explicit guard, ``True`` and ``False`` are accepted by ``float(...)`` and become
``1.0`` and ``0.0`` respectively, which silently changes match cardinality.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_matching_max_cost_validation_patch"
_ERROR_MESSAGE = "max_cost must be None or a finite non-negative scalar"


def install_matching_max_cost_validation(matching_module: Any) -> None:
    """Install idempotent validation for assignment max-cost gates."""

    original: Callable[..., Any] = matching_module.solve_bundle_linear_assignment
    if _method_chain_has_patch(original):
        return

    @wraps(original)
    def solve_bundle_linear_assignment_with_max_cost_validation(
        bundle: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "max_cost" in kwargs:
            normalized_kwargs = dict(kwargs)
            normalized_kwargs["max_cost"] = _normalize_max_cost(kwargs["max_cost"])
            kwargs = normalized_kwargs
        return original(bundle, *args, **kwargs)

    setattr(solve_bundle_linear_assignment_with_max_cost_validation, _PATCH_MARKER, True)
    setattr(
        solve_bundle_linear_assignment_with_max_cost_validation,
        "_bayescatrack_original",
        original,
    )
    matching_module.solve_bundle_linear_assignment = (  # type: ignore[assignment]
        solve_bundle_linear_assignment_with_max_cost_validation
    )


def _method_chain_has_patch(method: Any) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, "_bayescatrack_original", None)
    return False


def _normalize_max_cost(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(_ERROR_MESSAGE)

    try:
        value_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if value_array.shape != ():
        raise ValueError(_ERROR_MESSAGE)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(_ERROR_MESSAGE)
    try:
        numeric_value = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(_ERROR_MESSAGE)
    return numeric_value


__all__ = ["install_matching_max_cost_validation"]
