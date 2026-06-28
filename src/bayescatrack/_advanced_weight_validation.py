"""Strict validation for ROI pairwise-cost runtime knobs."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ._advanced_candidate_empty_validation import (
    install_advanced_candidate_empty_validation as _install_advanced_candidate_empty_validation,
)
from ._advanced_pruning_normalization import (
    install_advanced_pruning_normalization as _install_advanced_pruning_normalization,
)
from .core.bridge import CalciumPlaneData

_NONNEGATIVE_WEIGHT_KWARGS = (
    "centroid_weight",
    "iou_weight",
    "mask_cosine_weight",
    "area_weight",
    "roi_feature_weight",
    "cell_probability_weight",
    "radial_profile_weight",
    "orientation_weight",
    "eccentricity_weight",
    "compactness_weight",
    "border_proximity_weight",
    "ambiguity_margin_weight",
)
_POSITIVE_FLOAT_KWARGS = (
    "large_cost",
    "similarity_epsilon",
)
_OPTIONAL_POSITIVE_FLOAT_KWARGS = (
    "centroid_scale",
    "max_centroid_distance",
)
_BOOLEAN_KWARGS = (
    "shape_descriptor_components",
    "ambiguity_margin_components",
    "candidate_include_column_top_k",
)


def install_advanced_weight_validation() -> None:
    """Install idempotent validation around pairwise-cost kwargs."""

    _install_advanced_pruning_normalization()
    _install_advanced_candidate_empty_validation()
    _install_advanced_improvement_numeric_validation()

    original = CalciumPlaneData.build_pairwise_cost_matrix
    if _pairwise_method_chain_has_patch(
        original, "_bayescatrack_advanced_weight_validation_patch"
    ):
        return

    def _build_pairwise_cost_matrix_with_advanced_weight_validation(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        validated_kwargs = _validated_advanced_kwargs(kwargs)
        return original(self, other, *args, **validated_kwargs)

    setattr(
        _build_pairwise_cost_matrix_with_advanced_weight_validation,
        "_bayescatrack_advanced_weight_validation_patch",
        True,
    )
    setattr(
        _build_pairwise_cost_matrix_with_advanced_weight_validation,
        "_bayescatrack_original",
        original,
    )
    CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
        _build_pairwise_cost_matrix_with_advanced_weight_validation
    )


def _install_advanced_improvement_numeric_validation() -> None:
    validation = importlib.import_module(
        "bayescatrack.experiments._advanced_improvement_numeric_validation"
    )
    validation.install_advanced_improvement_numeric_validation()


def _validated_advanced_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    validated = dict(kwargs)
    for name in _NONNEGATIVE_WEIGHT_KWARGS:
        if name in validated:
            validated[name] = _finite_nonnegative_float(validated[name], name=name)
    for name in _POSITIVE_FLOAT_KWARGS:
        if name in validated:
            validated[name] = _finite_positive_float(validated[name], name=name)
    for name in _OPTIONAL_POSITIVE_FLOAT_KWARGS:
        if name in validated and validated[name] is not None:
            validated[name] = _finite_positive_float(validated[name], name=name)
    for name in _BOOLEAN_KWARGS:
        if name in validated:
            validated[name] = _strict_bool(validated[name], name=name)
    return validated


def _pairwise_method_chain_has_patch(method: Any, marker: str) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        seen.add(current_id)
        if getattr(current, marker, False):
            return True
        current = getattr(current, "_bayescatrack_original", None)
    return False


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive value") from exc
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return numeric_value


__all__ = ["install_advanced_weight_validation"]
