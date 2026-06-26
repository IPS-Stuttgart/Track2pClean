"""Strict runtime validation for experiment configuration values.

This module centralizes integer-like and finite-float validation for optional
configuration paths that are often exercised from YAML/CLI sweeps. The
validation is installed from :mod:`bayescatrack.__init__` so existing imports keep
using the public advanced-component module while rejecting ambiguous values such
as booleans, fractional top-k counts, NaN and infinity.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import advanced_roi_components as _advanced_roi_components

_PATCH_MARKER = "_bayescatrack_strict_config_validation_patch"
_WRAPPER_MARKER = "_bayescatrack_strict_config_validation_wrapper"
_ORIGINAL_ATTR = "_bayescatrack_strict_config_original"


@dataclass(frozen=True)
class CandidatePruningConfig:
    """Top-k candidate pruning configuration for pairwise ROI costs."""

    top_k_per_roi: int | None = None
    include_column_top_k: bool = True
    gate_margin: float | None = None
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        if self.top_k_per_roi is not None:
            object.__setattr__(
                self,
                "top_k_per_roi",
                _positive_int(self.top_k_per_roi, name="top_k_per_roi"),
            )
        object.__setattr__(
            self,
            "include_column_top_k",
            _strict_bool(self.include_column_top_k, name="include_column_top_k"),
        )
        if self.gate_margin is not None:
            object.__setattr__(
                self,
                "gate_margin",
                _finite_nonnegative_float(self.gate_margin, name="gate_margin"),
            )
        object.__setattr__(
            self,
            "large_cost",
            _finite_positive_float(self.large_cost, name="large_cost"),
        )


def install_strict_config_validation() -> None:
    """Install idempotent strict validation hooks for advanced components."""

    if getattr(_advanced_roi_components, _PATCH_MARKER, False) and _strict_hooks_active():
        return
    original_candidate_mask = _current_original(
        _advanced_roi_components.candidate_mask_from_cost_matrix
    )
    original_mask_shape_descriptors = _current_original(
        _advanced_roi_components.mask_shape_descriptors
    )
    setattr(candidate_mask_from_cost_matrix, _ORIGINAL_ATTR, original_candidate_mask)
    setattr(candidate_mask_from_cost_matrix, _WRAPPER_MARKER, True)
    setattr(mask_shape_descriptors, _ORIGINAL_ATTR, original_mask_shape_descriptors)
    setattr(mask_shape_descriptors, _WRAPPER_MARKER, True)

    _advanced_roi_components.CandidatePruningConfig = CandidatePruningConfig
    _advanced_roi_components.candidate_mask_from_cost_matrix = (
        candidate_mask_from_cost_matrix
    )
    _advanced_roi_components.mask_shape_descriptors = mask_shape_descriptors
    setattr(_advanced_roi_components, _PATCH_MARKER, True)


def candidate_mask_from_cost_matrix(
    cost_matrix: np.ndarray,
    *,
    top_k: int | None,
    include_columns: bool = True,
    gate_margin: float | None = None,
    large_cost: float = 1.0e6,
) -> np.ndarray:
    """Return a sparse candidate mask after strict scalar validation."""

    if top_k is not None:
        top_k = _positive_int(top_k, name="top_k")
    include_columns = _strict_bool(include_columns, name="include_columns")
    if gate_margin is not None:
        gate_margin = _finite_nonnegative_float(gate_margin, name="gate_margin")
    large_cost = _finite_positive_float(large_cost, name="large_cost")
    original = _original_function(
        candidate_mask_from_cost_matrix,
        "candidate_mask_from_cost_matrix",
    )
    return original(
        cost_matrix,
        top_k=top_k,
        include_columns=include_columns,
        gate_margin=gate_margin,
        large_cost=large_cost,
    )


def mask_shape_descriptors(
    masks: np.ndarray, *, radial_bins: int = 6
) -> dict[str, np.ndarray]:
    """Return per-ROI shape descriptors after validating ``radial_bins``."""

    radial_bins = _positive_int(radial_bins, name="radial_bins")
    original = _original_function(mask_shape_descriptors, "mask_shape_descriptors")
    return original(masks, radial_bins=radial_bins)


def _strict_hooks_active() -> bool:
    return (
        _advanced_roi_components.CandidatePruningConfig is CandidatePruningConfig
        and getattr(
            _advanced_roi_components.candidate_mask_from_cost_matrix,
            _WRAPPER_MARKER,
            False,
        )
        and getattr(
            _advanced_roi_components.mask_shape_descriptors,
            _WRAPPER_MARKER,
            False,
        )
    )


def _current_original(function: Callable[..., Any]) -> Callable[..., Any]:
    original = getattr(function, _ORIGINAL_ATTR, None)
    if original is None:
        return function
    return original


def _original_function(wrapper: Callable[..., Any], name: str) -> Callable[..., Any]:
    original = getattr(wrapper, _ORIGINAL_ATTR, None)
    if original is None:
        raise RuntimeError(
            f"strict config validation wrapper '{name}' is not installed"
        )
    return original


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(numeric_value)
    elif isinstance(value, bytes):
        raise ValueError(f"{name} must be an integer")
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(integer_value)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return numeric_value


__all__ = [
    "CandidatePruningConfig",
    "candidate_mask_from_cost_matrix",
    "install_strict_config_validation",
    "mask_shape_descriptors",
]
