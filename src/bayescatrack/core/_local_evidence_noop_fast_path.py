"""Fast-path no-op local-evidence pairwise-cost calls.

The local-evidence wrapper is zero-default: when no local-evidence term or
local-evidence diagnostics are requested, it should behave exactly like the
underlying pairwise-cost builder.  Avoid forcing the wrapper's internal
``return_components=True`` call in that no-op case so disabled diagnostic terms
remain disabled.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_local_evidence_noop_fast_path_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_LOCAL_EVIDENCE_ORIGINAL_FREEVAR = "original_build_pairwise_cost_matrix"

_LOCAL_EVIDENCE_WEIGHT_CONTROLS = (
    "weighted_dice_weight",
    "overlap_fraction_weight",
    "containment_weight",
    "distance_transform_weight",
    "image_patch_weight",
    "neighbor_constellation_weight",
    "centroid_rank_weight",
)
_LOCAL_EVIDENCE_ONLY_CONTROLS = frozenset(
    _LOCAL_EVIDENCE_WEIGHT_CONTROLS
    + (
        "local_evidence_components",
        "patch_radius",
        "neighbor_k",
        "normalize_weighted_overlap",
    )
)


def install_local_evidence_noop_fast_path(calcium_plane_cls: type[Any]) -> None:
    """Install an idempotent no-op bypass around the local-evidence wrapper."""

    local_evidence_method = calcium_plane_cls.build_pairwise_cost_matrix
    if _method_chain_has_patch(local_evidence_method):
        return

    base_method = _find_local_evidence_base_method(local_evidence_method)
    if base_method is None:
        return

    @wraps(local_evidence_method)
    def build_pairwise_cost_matrix_with_local_evidence_noop_fast_path(
        self: Any,
        other: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not args and _is_local_evidence_noop(kwargs):
            return base_method(
                self,
                other,
                **_drop_local_evidence_only_kwargs(kwargs),
            )
        return local_evidence_method(self, other, *args, **kwargs)

    setattr(
        build_pairwise_cost_matrix_with_local_evidence_noop_fast_path,
        _PATCH_MARKER,
        True,
    )
    setattr(
        build_pairwise_cost_matrix_with_local_evidence_noop_fast_path,
        _ORIGINAL_ATTR,
        local_evidence_method,
    )
    calcium_plane_cls.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        build_pairwise_cost_matrix_with_local_evidence_noop_fast_path
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
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


def _find_local_evidence_base_method(method: Any) -> Callable[..., Any] | None:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return None
        seen.add(current_id)

        base_method = _closure_value(
            current,
            name=_LOCAL_EVIDENCE_ORIGINAL_FREEVAR,
        )
        if callable(base_method):
            return base_method
        current = getattr(current, _ORIGINAL_ATTR, None)
    return None


def _closure_value(function: Any, *, name: str) -> Any | None:
    code = getattr(function, "__code__", None)
    closure = getattr(function, "__closure__", None)
    if code is None or closure is None:
        return None
    freevars = getattr(code, "co_freevars", ())
    try:
        index = freevars.index(name)
    except ValueError:
        return None
    if index >= len(closure):
        return None
    return closure[index].cell_contents


def _is_local_evidence_noop(kwargs: dict[str, Any]) -> bool:
    if _bool_control_is_true(kwargs.get("local_evidence_components", False)):
        return False
    return not any(
        _positive_weight(kwargs.get(weight_name, 0.0))
        for weight_name in _LOCAL_EVIDENCE_WEIGHT_CONTROLS
    )


def _drop_local_evidence_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key not in _LOCAL_EVIDENCE_ONLY_CONTROLS
    }


def _bool_control_is_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _positive_weight(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return True
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return True
    return np.isfinite(numeric) and numeric > 0.0


__all__ = ["install_local_evidence_noop_fast_path"]
