"""Validation patch for malformed advanced-uncertainty scalar controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import advanced_uncertainty as _advanced_uncertainty

# pylint: disable=protected-access

_PATCH_MARKER = "_bayescatrack_advanced_uncertainty_array_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_advanced_uncertainty_array_validation_original"
_POSTERIOR_PATCH_MARKER = "_bayescatrack_advanced_uncertainty_empty_probability_patch"
_POSTERIOR_ORIGINAL_ATTR = "_bayescatrack_advanced_uncertainty_empty_probability_original"
_STRING_LIKE_SCALAR_TYPES = (str, bytes, bytearray)


def install_advanced_uncertainty_array_validation() -> None:
    """Require numeric uncertainty/pruning controls and safe empty posteriors."""

    _install_scalar_control_validation()
    _install_empty_posterior_probability_validation()


def _install_scalar_control_validation() -> None:
    original = _advanced_uncertainty._validated_float
    if _method_chain_has_patch(
        original,
        patch_marker=_PATCH_MARKER,
        original_attr=_ORIGINAL_ATTR,
    ):
        return

    def validated_float(value: Any, *, name: str) -> float:
        if isinstance(value, _STRING_LIKE_SCALAR_TYPES):
            raise ValueError(f"{name} must be a numeric scalar")
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = value_array.item()
        if isinstance(scalar, _STRING_LIKE_SCALAR_TYPES):
            raise ValueError(f"{name} must be a numeric scalar")
        return original(scalar, name=name)

    validated_float.__name__ = original.__name__
    validated_float.__qualname__ = original.__qualname__
    setattr(validated_float, _PATCH_MARKER, True)
    setattr(validated_float, _ORIGINAL_ATTR, original)
    _advanced_uncertainty._validated_float = validated_float


def _install_empty_posterior_probability_validation() -> None:
    original = _advanced_uncertainty.posterior_probability_matrix
    if _method_chain_has_patch(
        original,
        patch_marker=_POSTERIOR_PATCH_MARKER,
        original_attr=_POSTERIOR_ORIGINAL_ATTR,
    ):
        return

    def posterior_probability_matrix(
        cost_matrix: Any,
        *,
        reliability_matrix: Any | None = None,
        temperature: float = 2.0,
    ) -> np.ndarray:
        costs = _advanced_uncertainty._as_cost_matrix(cost_matrix)
        if 0 not in costs.shape:
            return original(
                costs,
                reliability_matrix=reliability_matrix,
                temperature=temperature,
            )

        _advanced_uncertainty._validate_positive(temperature, name="temperature")
        if reliability_matrix is not None:
            reliability = np.asarray(reliability_matrix, dtype=float)
            if reliability.shape != costs.shape:
                raise ValueError("reliability_matrix must match cost_matrix shape")
        return np.zeros_like(costs, dtype=float)

    posterior_probability_matrix.__name__ = original.__name__
    posterior_probability_matrix.__qualname__ = original.__qualname__
    setattr(posterior_probability_matrix, _POSTERIOR_PATCH_MARKER, True)
    setattr(posterior_probability_matrix, _POSTERIOR_ORIGINAL_ATTR, original)
    _advanced_uncertainty.posterior_probability_matrix = posterior_probability_matrix


def _method_chain_has_patch(
    method: Any,
    *,
    patch_marker: str,
    original_attr: str,
) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, patch_marker, False):
            return True
        seen.add(current_id)
        current = getattr(current, original_attr, None)
    return False


__all__ = ["install_advanced_uncertainty_array_validation"]
