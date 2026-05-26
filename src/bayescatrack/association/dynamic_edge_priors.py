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
    """Additive edge-prior weights used before global assignment.

    The reciprocal-rank terms are a label-free association prior: confident
    low-cost links should usually be among the best candidates from both the
    source ROI and the target ROI perspectives.  A small relief on such links
    helps preserve complete tracks, while a small penalty on one-sided top
    candidates discourages many-to-one/one-to-many ambiguities.
    """

    session_gap_weight: float = 0.0
    cell_probability_weight: float = 0.0
    area_ratio_weight: float = 0.0
    activity_missing_weight: float = 0.0
    registration_empty_roi_weight: float = 0.0
    edge_quality_bias: float = 0.0
    reciprocal_rank_relief: float = 0.0
    reciprocal_rank_penalty: float = 0.0
    reciprocal_rank_max_rank: int = 1
    reciprocal_rank_min_margin: float = 0.0
    reciprocal_rank_consecutive_only: bool = False
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        for name in (
            "session_gap_weight",
            "cell_probability_weight",
            "area_ratio_weight",
            "activity_missing_weight",
            "registration_empty_roi_weight",
            "reciprocal_rank_relief",
            "reciprocal_rank_penalty",
            "reciprocal_rank_min_margin",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "edge_quality_bias", float(self.edge_quality_bias))
        max_rank = int(self.reciprocal_rank_max_rank)
        if max_rank < 1:
            raise ValueError("reciprocal_rank_max_rank must be at least 1")
        object.__setattr__(self, "reciprocal_rank_max_rank", max_rank)
        object.__setattr__(
            self,
            "reciprocal_rank_consecutive_only",
            bool(self.reciprocal_rank_consecutive_only),
        )
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
        empty = _column_mask_for_cost_shape(empty_registered_rois, costs.shape)
        costs[:, empty] += cfg.registration_empty_roi_weight
    if _reciprocal_rank_prior_is_enabled(cfg) and not (
        cfg.reciprocal_rank_consecutive_only and int(session_gap) != 1
    ):
        costs = _apply_reciprocal_rank_prior(costs, cfg)

    return np.nan_to_num(
        costs,
        nan=cfg.large_cost,
        posinf=cfg.large_cost,
        neginf=cfg.large_cost,
    )


def _reciprocal_rank_prior_is_enabled(cfg: DynamicEdgePriorConfig) -> bool:
    return bool(cfg.reciprocal_rank_relief or cfg.reciprocal_rank_penalty)


def _apply_reciprocal_rank_prior(
    cost_matrix: np.ndarray,
    cfg: DynamicEdgePriorConfig,
) -> np.ndarray:
    """Reward mutual low-rank edges and penalize one-sided top candidates."""

    costs = np.asarray(cost_matrix, dtype=float).copy()
    if costs.size == 0:
        return costs
    admissible = np.isfinite(costs) & (costs < float(cfg.large_cost))
    if not np.any(admissible):
        return costs

    row_ranks = _axis_cost_ranks(costs, admissible=admissible, axis=1)
    column_ranks = _axis_cost_ranks(costs, admissible=admissible, axis=0)
    top_by_row = row_ranks <= int(cfg.reciprocal_rank_max_rank)
    top_by_column = column_ranks <= int(cfg.reciprocal_rank_max_rank)
    reciprocal = admissible & top_by_row & top_by_column

    if cfg.reciprocal_rank_min_margin > 0.0:
        row_margins = _axis_next_cost_margins(costs, admissible=admissible, axis=1)
        column_margins = _axis_next_cost_margins(
            costs,
            admissible=admissible,
            axis=0,
        )
        reciprocal &= np.minimum(row_margins, column_margins) >= float(
            cfg.reciprocal_rank_min_margin
        )

    one_sided_top = admissible & (top_by_row | top_by_column) & ~reciprocal
    if cfg.reciprocal_rank_penalty:
        costs[one_sided_top] += float(cfg.reciprocal_rank_penalty)
    if cfg.reciprocal_rank_relief:
        costs[reciprocal] -= float(cfg.reciprocal_rank_relief)
    return costs


def _axis_cost_ranks(
    costs: np.ndarray,
    *,
    admissible: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Return one-based stable ranks along rows or columns, with large sentinel ranks."""

    ranks = np.full(costs.shape, np.iinfo(np.int32).max, dtype=np.int32)
    if axis == 1:
        for row_index in range(costs.shape[0]):
            indices = np.flatnonzero(admissible[row_index, :])
            if indices.size == 0:
                continue
            ordered = indices[np.argsort(costs[row_index, indices], kind="stable")]
            ranks[row_index, ordered] = np.arange(1, ordered.size + 1, dtype=np.int32)
        return ranks
    if axis == 0:
        for column_index in range(costs.shape[1]):
            indices = np.flatnonzero(admissible[:, column_index])
            if indices.size == 0:
                continue
            ordered = indices[np.argsort(costs[indices, column_index], kind="stable")]
            ranks[ordered, column_index] = np.arange(1, ordered.size + 1, dtype=np.int32)
        return ranks
    raise ValueError("axis must be 0 or 1")


def _axis_next_cost_margins(
    costs: np.ndarray,
    *,
    admissible: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Return next-worse-candidate cost margins along rows or columns."""

    margins = np.full(costs.shape, np.inf, dtype=float)
    if axis == 1:
        for row_index in range(costs.shape[0]):
            indices = np.flatnonzero(admissible[row_index, :])
            if indices.size <= 1:
                continue
            ordered = indices[np.argsort(costs[row_index, indices], kind="stable")]
            ordered_costs = costs[row_index, ordered]
            margins[row_index, ordered[:-1]] = ordered_costs[1:] - ordered_costs[:-1]
        return margins
    if axis == 0:
        for column_index in range(costs.shape[1]):
            indices = np.flatnonzero(admissible[:, column_index])
            if indices.size <= 1:
                continue
            ordered = indices[np.argsort(costs[indices, column_index], kind="stable")]
            ordered_costs = costs[ordered, column_index]
            margins[ordered[:-1], column_index] = ordered_costs[1:] - ordered_costs[:-1]
        return margins
    raise ValueError("axis must be 0 or 1")


def _component(
    pairwise_components: Mapping[str, Any], component_name: str, shape: tuple[int, int]
) -> np.ndarray:
    if component_name not in pairwise_components:
        return np.zeros(shape, dtype=float)
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.shape != shape:
        raise ValueError(f"Pairwise component {component_name!r} has wrong shape")
    return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=0.0)


def _activity_missing_component(pairwise_components: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray:
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
