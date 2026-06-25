"""Strict validation for pairwise-cost boolean controls.

Pairwise-cost wrappers inspect boolean flags before the base bridge method sees
them.  Rejecting ambiguous values keeps pairwise-cost return and IoU behavior
explicit.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_pairwise_bool_control_validation_patch"
_BOOL_CONTROL_NAMES = ("return_components", "soft_iou")


def install_return_components_validation(bridge_module: Any) -> None:
    """Install idempotent validation for pairwise-cost boolean controls."""

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
        normalized_kwargs: dict[str, Any] | None = None
        for control_name in _BOOL_CONTROL_NAMES:
            if control_name in kwargs:
                if normalized_kwargs is None:
                    normalized_kwargs = dict(kwargs)
                normalized_kwargs[control_name] = _strict_bool(
                    kwargs[control_name], name=control_name
                )
        if normalized_kwargs is not None:
            kwargs = normalized_kwargs
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


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


__all__ = ["install_return_components_validation"]
