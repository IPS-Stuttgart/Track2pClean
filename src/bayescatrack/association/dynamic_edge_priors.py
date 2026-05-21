"""Dynamic pairwise edge priors for Track2p global assignment.

PyRecEst currently receives scalar start/end/gap costs from BayesCaTrack.  This
module provides a conservative bridge toward learned/non-uniform priors by
adding optional ROI-, edge-, and session-gap-dependent penalties directly to the
pairwise cost matrices before global assignment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DynamicEdgePriorConfig:
    """Additive edge-prior weights used before global assignment."""

    session_gap_weight: float = 0.0
    cell_probability_weight: float = 0.0
    area_ratio_weight: float = 0.0
    activity_missing_weight: float = 0.0
    registration_empty_roi_weight: float = 0.0
    edge_quality_bias: float = 0.0
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        for name in (
            "session_gap_weight",
            "cell_probability_weight",
            "area_ratio_weight",
            "activity_missing_weight",
            "registration_empty_roi_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "edge_quality_bias", float(self.edge_quality_bias))
        large_cost = float(self.large_cost)
        if not np.isfinite(large_cost) or large_cost <= 0.0:
            raise ValueError("large_cost must be finite and positive")
        object.__setattr__(self, "large_cost", large_cost)


def dynamic_edge_prior_config_from_mapping(
    value: DynamicEdgePriorConfig | Mapping[str, Any] | None,
) -> DynamicEdgePriorConfig | None:
    if value is None:
        return None
    if isinstance(value, DynamicEdgePriorConfig):
        return value
    return DynamicEdgePriorConfig(**dict(value))


def apply_dynamic_edge_priors(
    cost_matrix: Any,
    pairwise_components: Mapping[str, Any],
    *,
    session_gap: int | float,
    empty_registered_rois: Any | None = None,
    config: DynamicEdgePriorConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return costs with optional edge-prior penalties added."""

    costs = np.asarray(cost_matrix, dtype=float).copy()
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    cfg = dynamic_edge_prior_config_from_mapping(config)
    if cfg is None:
        return costs

    if cfg.edge_quality_bias:
        costs += float(cfg.edge_quality_bias)
    if cfg.session_gap_weight:
        costs += cfg.session_gap_weight * max(float(session_gap) - 1.0, 0.0)
    if cfg.cell_probability_weight:
        costs += cfg.cell_probability_weight * _component(
            pairwise_components,
            "cell_probability_cost",
            costs.shape,
        )
    if cfg.area_ratio_weight:
        costs += cfg.area_ratio_weight * _component(
            pairwise_components,
            "area_ratio_cost",
            costs.shape,
        )
    if cfg.activity_missing_weight:
        missing = _activity_missing_component(pairwise_components, costs.shape)
        costs += cfg.activity_missing_weight * missing
    if cfg.registration_empty_roi_weight and empty_registered_rois is not None:
        empty = np.asarray(empty_registered_rois, dtype=bool).reshape(-1)
        if empty.shape != (costs.shape[1],):
            non_empty_count = int(empty.size - np.count_nonzero(empty))
            if non_empty_count != costs.shape[1]:
                raise ValueError(
                    "empty_registered_rois must describe either the current cost "
                    "matrix columns or the full registered-ROI column layout"
                )
            # The cost matrix is already compacted to non-empty registered ROI
            # columns. Empty ROI columns will be reintroduced downstream by
            # expand_registered_pairwise_cost_columns(..., fill_value=large_cost),
            # so there is no compact column to penalize here.
        else:
            costs[:, empty] += cfg.registration_empty_roi_weight

    return np.nan_to_num(
        costs,
        nan=cfg.large_cost,
        posinf=cfg.large_cost,
        neginf=cfg.large_cost,
    )


def _component(
    pairwise_components: Mapping[str, Any], component_name: str, shape: tuple[int, int]
) -> np.ndarray:
    if component_name not in pairwise_components:
        return np.zeros(shape, dtype=float)
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.shape != shape:
        raise ValueError(f"Pairwise component {component_name!r} has wrong shape")
    return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=0.0)


def _activity_missing_component(
    pairwise_components: Mapping[str, Any], shape: tuple[int, int]
) -> np.ndarray:
    for name in (
        "activity_tiebreaker_missing",
        "activity_similarity_available",
        "fluorescence_similarity_available",
        "spike_similarity_available",
    ):
        if name not in pairwise_components:
            continue
        values = np.asarray(pairwise_components[name], dtype=float)
        if values.shape != shape:
            continue
        if name.endswith("_available"):
            return 1.0 - np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
        return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    return np.zeros(shape, dtype=float)


__all__ = (
    "DynamicEdgePriorConfig",
    "apply_dynamic_edge_priors",
    "dynamic_edge_prior_config_from_mapping",
)
