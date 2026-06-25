"""Strict validation for pairwise-cost return-component controls.

Pairwise-cost wrappers dispatch on ``return_components`` before the base bridge
method sees it.  Relying on Python truthiness lets malformed values such as
``"false"`` or ``1`` silently alter the return type.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_return_components_validation_patch"
_ERROR_MESSAGE = "return_components must be a boolean"


def install_return_components_validation(bridge_module: Any) -> None:
    """Install idempotent validation for pairwise-cost return controls."""

    original = bridge_module.CalciumPlaneData.build_pairwise_cost_matrix
    if _method_chain_has_patch(original):
        return

    @wraps(original)
    def build_pairwise_cost_matrix_with_return_components_validation(
        self: Any,
        other: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "return_components" in kwargs:
            kwargs = dict(kwargs)
            kwargs["return_components"] = _strict_bool(kwargs["return_components"])
        return original(self, other, *args, **kwargs)

    setattr(
        build_pairwise_cost_matrix_with_return_components_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        build_pairwise_cost_matrix_with_return_components_validation,
        "_bayescatrack_original",
        original,
    )
    bridge_module.CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        build_pairwise_cost_matrix_with_return_components_validation
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


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(_ERROR_MESSAGE)


__all__ = ["install_return_components_validation"]
