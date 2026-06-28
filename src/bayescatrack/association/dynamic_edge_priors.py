"""Dynamic pairwise edge priors for Track2p global assignment.

PyRecEst currently receives scalar start/end/gap costs from BayesCaTrack.  This
module provides a conservative bridge toward learned/non-uniform priors by
adding optional ROI-, edge-, and session-gap-dependent penalties directly to the
pairwise cost matrices before global assignment.
"""

from __future__ import annotations

import operator
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
    local_margin_weight: float = 0.0
    local_margin_target: float = 0.0
    local_margin_cap: float | None = None
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
            "local_margin_weight",
            "local_margin_target",
        ):
            value = _validated_non_negative_finite_float(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if self.reciprocal_rank_cap is not None:
            reciprocal_rank_cap = _validated_non_negative_finite_float(
                "reciprocal_rank_cap",
                self.reciprocal_rank_cap,
            )
            object.__setattr__(self, "reciprocal_rank_cap", reciprocal_rank_cap)
        if self.local_margin_cap is not None:
            local_margin_cap = _validated_non_negative_finite_float(
                "local_margin_cap",
                self.local_margin_cap,
            )
            object.__setattr__(self, "local_margin_cap", local_margin_cap)
        edge_quality_bias = _validated_finite_float(
            "edge_quality_bias",
            self.edge_quality_bias,
        )
        object.__setattr__(self, "edge_quality_bias", edge_quality_bias)
        large_cost = _validated_positive_finite_float("large_cost", self.large_cost)
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
            cfg.session_gap_weight * _validated_session_gap_offset(session_gap),
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
    if cfg.local_margin_weight and cfg.local_margin_target:
        costs += _local_margin_shortfall_penalty(
            costs,
            weight=cfg.local_margin_weight,
            target_margin=cfg.local_margin_target,
            cap=cfg.local_margin_cap,
            large_cost=cfg.large_cost,
        )
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


def _validated_session_gap_offset(session_gap: int | float) -> float:
    gap = _validated_positive_integer_session_gap(session_gap)
    return float(gap - 1)


def _validated_positive_integer_session_gap(session_gap: Any) -> int:
    message = (
        "session_gap must be a finite value representing an integer "
        "greater than or equal to 1"
    )
    if isinstance(session_gap, (bool, np.bool_)):
        raise ValueError(message)
    try:
        session_gap_array = np.asarray(session_gap)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if session_gap_array.shape != ():
        raise ValueError(message)
    if isinstance(session_gap_array.item(), (bool, np.bool_)):
        raise ValueError(message)

    try:
        gap = int(operator.index(session_gap))
    except TypeError:
        if isinstance(session_gap, str):
            text = session_gap.strip()
            if not text:
                raise ValueError(message) from None
            try:
                numeric_gap = float(text)
            except ValueError as exc:
                raise ValueError(message) from exc
        else:
            try:
                numeric_gap = float(session_gap)
            except (TypeError, ValueError) as exc:
                raise ValueError(message) from exc
        if not np.isfinite(numeric_gap) or not numeric_gap.is_integer():
            raise ValueError(message)
        gap = int(numeric_gap)

    if gap < 1:
        raise ValueError(message)
    return gap


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


def _local_margin_shortfall_penalty(
    costs: np.ndarray,
    *,
    weight: float,
    target_margin: float,
    cap: float | None,
    large_cost: float,
) -> np.ndarray:
    """Return penalties for locally best edges with weak separation.

    Dense reciprocal rank penalizes non-competitive alternatives but leaves all
    local best edges unchanged.  Ambiguous Track2p false continuations can still
    be row/column best when the local field contains several almost-equal
    candidates.  This prior adds a bounded shortfall penalty only when a valid
    edge is best along the row or column and its nearest alternative is closer
    than ``target_margin``.
    """

    if weight <= 0.0 or target_margin <= 0.0:
        return np.zeros_like(costs, dtype=float)
    row_margins = _best_axis_margins(costs, axis=1, large_cost=large_cost)
    column_margins = _best_axis_margins(costs, axis=0, large_cost=large_cost)
    local_margin = np.minimum(row_margins, column_margins)
    finite = np.isfinite(local_margin)
    penalty = np.zeros_like(costs, dtype=float)
    if not np.any(finite):
        return penalty
    penalty[finite] = float(weight) * np.maximum(
        float(target_margin) - local_margin[finite], 0.0
    )
    if cap is not None:
        penalty[finite] = np.minimum(penalty[finite], float(cap))
    return penalty


def _best_axis_margins(
    costs: np.ndarray, *, axis: int, large_cost: float
) -> np.ndarray:
    margins = np.full(costs.shape, np.inf, dtype=float)
    if costs.size == 0:
        return margins
    if axis == 1:
        for row_index in range(costs.shape[0]):
            margins[row_index, :] = _best_vector_margins(
                costs[row_index, :], large_cost=large_cost
            )
        return margins
    if axis == 0:
        for column_index in range(costs.shape[1]):
            margins[:, column_index] = _best_vector_margins(
                costs[:, column_index], large_cost=large_cost
            )
        return margins
    raise ValueError("axis must be 0 or 1")


def _best_vector_margins(values: np.ndarray, *, large_cost: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    margins = np.full(vector.shape, np.inf, dtype=float)
    valid = np.isfinite(vector) & (vector < large_cost)
    if not np.any(valid):
        return margins.reshape(values.shape)
    valid_values = vector[valid]
    best = float(np.min(valid_values))
    best_mask = valid & np.isclose(vector, best, rtol=1.0e-12, atol=1.0e-12)
    if np.count_nonzero(valid) <= 1:
        return margins.reshape(values.shape)
    if np.count_nonzero(best_mask) > 1:
        margins[best_mask] = 0.0
        return margins.reshape(values.shape)
    alternative_mask = ~np.isclose(valid_values, best, rtol=1.0e-12, atol=1.0e-12)
    alternatives = valid_values[alternative_mask]
    if alternatives.size == 0:
        margins[best_mask] = 0.0
    else:
        margins[best_mask] = float(np.min(alternatives) - best)
    return margins.reshape(values.shape)


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
    """Return a validated column mask aligned to a compact or full cost matrix.

    Registered ROI masks are often computed in the original measurement-ROI
    layout, while costs are built on a compact non-empty subset and expanded only
    after pruning.  If the supplied mask is already compact, use it directly. If
    it is full-size and the compact matrix has one column per non-empty ROI,
    there are no empty columns left to penalize at this stage.
    """

    column_mask = _strict_binary_column_mask(mask, name="empty_registered_rois")
    if column_mask.shape == (shape[1],):
        return column_mask
    compact_column_count = int(column_mask.size - np.count_nonzero(column_mask))
    if compact_column_count == shape[1]:
        return np.zeros((shape[1],), dtype=bool)
    raise ValueError("empty_registered_rois must align with compact or full columns")


def _strict_binary_column_mask(mask: Any, *, name: str) -> np.ndarray:
    message = f"{name} must be a boolean or binary numeric mask"
    try:
        values = np.asarray(mask).reshape(-1)
    except ValueError as exc:
        raise ValueError(message) from exc

    if values.dtype.kind == "b":
        return np.ascontiguousarray(values, dtype=bool)
    if values.dtype.kind in {"i", "u"}:
        if np.any((values != 0) & (values != 1)):
            raise ValueError(message)
        return np.ascontiguousarray(values.astype(bool, copy=False), dtype=bool)
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
            raise ValueError(message)
        return np.ascontiguousarray(values.astype(bool, copy=False), dtype=bool)
    raise ValueError(message)


def _validated_non_negative_finite_float(name: str, raw_value: Any) -> float:
    value = _validated_numeric_scalar(name, raw_value)
    if value < 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validated_finite_float(name: str, raw_value: Any) -> float:
    value = _validated_numeric_scalar(name, raw_value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validated_positive_finite_float(name: str, raw_value: Any) -> float:
    value = _validated_numeric_scalar(name, raw_value)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validated_numeric_scalar(name: str, raw_value: Any) -> float:
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric scalar, not a boolean")
    if isinstance(raw_value, str):
        raise ValueError(f"{name} must be a numeric scalar, not text")

    raw_array = np.asarray(raw_value)
    if raw_array.ndim != 0:
        raise ValueError(f"{name} must be a numeric scalar")
    raw_scalar = raw_array.item()
    if isinstance(raw_scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric scalar, not a boolean")
    if isinstance(raw_scalar, str):
        raise ValueError(f"{name} must be a numeric scalar, not text")
    try:
        return float(raw_scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric scalar") from exc


__all__ = (
    "DynamicEdgePriorConfig",
    "apply_dynamic_edge_priors",
    "dynamic_edge_prior_config_from_mapping",
)
