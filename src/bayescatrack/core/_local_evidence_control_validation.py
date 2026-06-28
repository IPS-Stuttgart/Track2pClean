"""Strict validation for local-evidence pairwise-cost controls.

The local-evidence cost wrapper consumes several optional controls before the
base core scalar validators see them.  Normalize those controls at the wrapper
boundary so NumPy singleton arrays, boolean-like numeric values, or strings
cannot be silently coerced into weights, radii, or flags.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_local_evidence_control_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"

_FLOAT_CONTROLS = (
    "weighted_dice_weight",
    "overlap_fraction_weight",
    "containment_weight",
    "distance_transform_weight",
    "image_patch_weight",
    "neighbor_constellation_weight",
    "centroid_rank_weight",
)
_BOOLEAN_CONTROLS = (
    "local_evidence_components",
    "normalize_weighted_overlap",
    "return_components",
)
_INTEGER_CONTROLS = {
    "patch_radius": (0, "non-negative"),
    "neighbor_k": (1, "at least 1"),
}
_STRINGLIKE_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_local_evidence_control_validation(calcium_plane_cls: type[Any]) -> None:
    """Install idempotent validation for local-evidence pairwise controls."""

    original = calcium_plane_cls.build_pairwise_cost_matrix
    if _method_chain_has_patch(original):
        return

    @wraps(original)
    def build_pairwise_cost_matrix_with_local_evidence_control_validation(
        self: Any,
        other: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            kwargs = _validate_local_evidence_kwargs(kwargs)
        return original(self, other, *args, **kwargs)

    setattr(
        build_pairwise_cost_matrix_with_local_evidence_control_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        build_pairwise_cost_matrix_with_local_evidence_control_validation,
        _ORIGINAL_ATTR,
        original,
    )
    calcium_plane_cls.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        build_pairwise_cost_matrix_with_local_evidence_control_validation
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


def _validate_local_evidence_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    validated = dict(kwargs)
    for name in _FLOAT_CONTROLS:
        if name in validated:
            validated[name] = _finite_nonnegative_float(validated[name], name=name)
    for name in _BOOLEAN_CONTROLS:
        if name in validated:
            validated[name] = _strict_bool(validated[name], name=name)
    for name, (minimum, minimum_message) in _INTEGER_CONTROLS.items():
        if name in validated:
            validated[name] = _integer_control(
                validated[name],
                name=name,
                minimum=minimum,
                minimum_message=minimum_message,
            )
    return validated


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    message = f"{name} must be a finite non-negative value"
    value = _unwrap_scalar_array(value, message=message)
    if isinstance(value, (bool, np.bool_)) or isinstance(value, _STRINGLIKE_SCALAR_TYPES):
        raise ValueError(message)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(message)
    return numeric_value


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _integer_control(
    value: Any,
    *,
    name: str,
    minimum: int,
    minimum_message: str,
) -> int:
    message = f"{name} must be an integer"
    value = _unwrap_scalar_array(value, message=message)
    if isinstance(value, (bool, np.bool_)) or isinstance(value, _STRINGLIKE_SCALAR_TYPES):
        raise ValueError(message)
    try:
        integer_value = operator.index(value)
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(message) from None
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(message)
        integer_value = int(numeric_value)

    integer_value = int(integer_value)
    if integer_value < minimum:
        raise ValueError(f"{name} must be {minimum_message}")
    return integer_value


def _unwrap_scalar_array(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(message)
        return value.item()
    return value


__all__ = ["install_local_evidence_control_validation"]
