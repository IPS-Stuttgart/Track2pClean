"""Strict validation for association-bundle boolean controls.

Association-bundle builders use ``weighted_centroids`` to select weighted ROI
moments and ``return_pairwise_components`` to choose the diagnostic output path.
Accepting arbitrary truthy/falsy values would let strings or integers silently
change tracking evidence and diagnostics. This hook keeps those controls
explicit at the public bridge boundary.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_association_bundle_bool_validation_patch"
_BOOL_CONTROL_NAMES = ("weighted_centroids", "return_pairwise_components")
_PAIRWISE_COMPONENT_RETURN_KEY = "return_components"


def install_association_bundle_bool_validation(bridge_impl: Any) -> None:
    """Install idempotent validation around association-bundle builders."""

    for function_name in (
        "build_session_pair_association_bundle",
        "build_consecutive_session_association_bundles",
    ):
        _patch_builder(bridge_impl, function_name)


def _patch_builder(bridge_impl: Any, function_name: str) -> None:
    original = getattr(bridge_impl, function_name)
    if _function_chain_has_patch(original):
        return

    @wraps(original)
    def builder_with_bool_validation(*args: Any, **kwargs: Any) -> Any:
        normalized_kwargs: dict[str, Any] | None = None
        for control_name in _BOOL_CONTROL_NAMES:
            if control_name in kwargs:
                if normalized_kwargs is None:
                    normalized_kwargs = dict(kwargs)
                normalized_kwargs[control_name] = _strict_bool(
                    kwargs[control_name], name=control_name
                )

        if "pairwise_cost_kwargs" in kwargs and kwargs["pairwise_cost_kwargs"] is not None:
            normalized_pairwise_cost_kwargs = dict(kwargs["pairwise_cost_kwargs"])
            if _PAIRWISE_COMPONENT_RETURN_KEY in normalized_pairwise_cost_kwargs:
                if normalized_kwargs is None:
                    normalized_kwargs = dict(kwargs)
                normalized_pairwise_cost_kwargs.pop(_PAIRWISE_COMPONENT_RETURN_KEY, None)
                normalized_kwargs["pairwise_cost_kwargs"] = normalized_pairwise_cost_kwargs

        if normalized_kwargs is not None:
            kwargs = normalized_kwargs
        return original(*args, **kwargs)

    setattr(builder_with_bool_validation, _PATCH_MARKER, True)
    setattr(builder_with_bool_validation, "_bayescatrack_original", original)
    setattr(bridge_impl, function_name, builder_with_bool_validation)


def _function_chain_has_patch(function: Any) -> bool:
    seen: set[int] = set()
    current: Any = function
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


__all__ = ["install_association_bundle_bool_validation"]