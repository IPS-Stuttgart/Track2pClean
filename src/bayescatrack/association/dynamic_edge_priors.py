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
    reciprocal_rank_weight: float = 0.0
    reciprocal_rank_cap: float | None = None
    edge_quality_bias: float = 0.0
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        for name in (
            "session_gap_weight",
            "cell_probability_weight",
            "area_ratio_weight",
            "activity_missing_weight",
            "registration_empty_roi_weight",
            "reciprocal_rank_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.reciprocal_rank_cap is not None:
            reciprocal_rank_cap = float(self.reciprocal_rank_cap)
            if reciprocal_rank_cap < 0.0 or not np.isfinite(reciprocal_rank_cap):
                raise ValueError("reciprocal_rank_cap must be finite and non-negative")
            object.__setattr__(self, "reciprocal_rank_cap", reciprocal_rank_cap)
        edge_quality_bias = float(self.edge_quality_bias)
        if not np.isfinite(edge_quality_bias):
            raise ValueError("edge_quality_bias must be finite")
        object.__setattr__(self, "edge_quality_bias", edge_quality_bias)
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

    valid_edge_mask = _valid_edge_mask(costs, large_cost=cfg.large_cost)

    if cfg.edge_quality_bias:
        _add_to_valid_edges(costs, float(cfg.edge_quality_bias), valid_edge_mask)
    if cfg.session_gap_weight:
        _add_to_valid_edges(
            costs,
            cfg.session_gap_weight * max(float(session_gap) - 1.0, 0.0),
            valid_edge_mask,
        )
    if cfg.cell_probability_weight:
        _add_to_valid_edges(
            costs,
            cfg.cell_probability_weight
            * _component(
                pairwise_components,
                "cell_probability_cost",
                costs.shape,
            ),
            valid_edge_mask,
        )
    if cfg.area_ratio_weight:
        _add_to_valid_edges(
            costs,
            cfg.area_ratio_weight
            * _component(
                pairwise_components,
                "area_ratio_cost",
                costs.shape,
            ),
            valid_edge_mask,
        )
    if cfg.activity_missing_weight:
        missing = _activity_missing_component(pairwise_components, costs.shape)
        _add_to_valid_edges(
            costs, cfg.activity_missing_weight * missing, valid_edge_mask
        )
    if cfg.registration_empty_roi_weight and empty_registered_rois is not None:
        empty = _column_mask_for_cost_shape(empty_registered_rois, costs.shape)
        empty_columns = np.broadcast_to(empty[None, :], costs.shape)
        costs[valid_edge_mask & empty_columns] += cfg.registration_empty_roi_weight
    if cfg.reciprocal_rank_weight:
        costs += _reciprocal_rank_penalty(
            costs,
            weight=cfg.reciprocal_rank_weight,
            cap=cfg.reciprocal_rank_cap,
            large_cost=cfg.large_cost,
        )

    adjusted = np.nan_to_num(
        costs,
        nan=cfg.large_cost,
        posinf=cfg.large_cost,
        neginf=cfg.large_cost,
    )
    adjusted[~valid_edge_mask] = cfg.large_cost
    return adjusted


def _valid_edge_mask(costs: np.ndarray, *, large_cost: float) -> np.ndarray:
    return np.isfinite(costs) & (costs < float(large_cost))


def _add_to_valid_edges(
    costs: np.ndarray,
    increment: float | np.ndarray,
    valid_edge_mask: np.ndarray,
) -> None:
    if np.isscalar(increment):
        costs[valid_edge_mask] += float(increment)
        return

    values = np.asarray(increment, dtype=float)
    if values.shape != costs.shape:
        raise ValueError("edge-prior increment must be scalar or match cost shape")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    costs[valid_edge_mask] += values[valid_edge_mask]


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


def _reciprocal_rank_penalty(
    costs: np.ndarray,
    *,
    weight: float,
    cap: float | None,
    large_cost: float,
) -> np.ndarray:
    """Return a log-scaled penalty for links that are not row/column competitive.

    Costs are better when smaller.  A mutually best edge has row rank 0 and
    column rank 0, so it receives no penalty.  Ambiguous alternatives receive a
    penalty proportional to the worse of their row and column dense ranks.
    """

    if weight <= 0.0:
        return np.zeros_like(costs, dtype=float)
    row_ranks = _dense_axis_ranks(costs, axis=1, large_cost=large_cost)
    column_ranks = _dense_axis_ranks(costs, axis=0, large_cost=large_cost)
    finite = np.isfinite(row_ranks) & np.isfinite(column_ranks)
    penalty = np.zeros_like(costs, dtype=float)
    if not np.any(finite):
        return penalty
    reciprocal_rank = np.maximum(row_ranks, column_ranks)
    penalty[finite] = float(weight) * np.log1p(reciprocal_rank[finite])
    if cap is not None:
        penalty[finite] = np.minimum(penalty[finite], float(cap))
    return penalty


def _dense_axis_ranks(costs: np.ndarray, *, axis: int, large_cost: float) -> np.ndarray:
    ranks = np.full(costs.shape, np.inf, dtype=float)
    if costs.size == 0:
        return ranks
    if axis == 1:
        for row_index in range(costs.shape[0]):
            ranks[row_index, :] = _dense_vector_ranks(
                costs[row_index, :], large_cost=large_cost
            )
        return ranks
    if axis == 0:
        for column_index in range(costs.shape[1]):
            ranks[:, column_index] = _dense_vector_ranks(
                costs[:, column_index], large_cost=large_cost
            )
        return ranks
    raise ValueError("axis must be 0 or 1")


def _dense_vector_ranks(values: np.ndarray, *, large_cost: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    ranks = np.full(vector.shape, np.inf, dtype=float)
    valid = np.isfinite(vector) & (vector < large_cost)
    if not np.any(valid):
        return ranks
    valid_values = vector[valid]
    unique_values = np.unique(valid_values)
    ranks[valid] = np.searchsorted(unique_values, valid_values).astype(float)
    return ranks.reshape(values.shape)


def _column_mask_for_cost_shape(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    """Return a column mask aligned to a compact or full cost matrix.

    Registered ROI masks are often computed in the original measurement-ROI
    layout, while costs are built on a compact non-empty subset and expanded only
    after pruning.  If the supplied mask is already compact, use it directly. If
    it is full-size and the compact matrix has one column per non-empty ROI,
    there are no empty columns left to penalize at this stage.
    """

    column_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if column_mask.shape == (shape[1],):
        return column_mask
    compact_column_count = int(column_mask.size - np.count_nonzero(column_mask))
    if compact_column_count == shape[1]:
        return np.zeros((shape[1],), dtype=bool)
    raise ValueError("empty_registered_rois must align with compact or full columns")


__all__ = (
    "DynamicEdgePriorConfig",
    "apply_dynamic_edge_priors",
    "dynamic_edge_prior_config_from_mapping",
)
