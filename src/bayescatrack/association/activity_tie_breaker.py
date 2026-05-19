"""Weak, centered activity tie-breaker costs for ROI association."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def activity_tie_breaker_cost_matrix(
    pairwise_components: Mapping[str, Any],
    *,
    component_name: str = "activity_tiebreaker_cost",
    weight: float = 0.05,
    availability_component: str | None = "activity_tiebreaker_available",
    neutral_cost: float = 0.5,
    base_cost_matrix: Any | None = None,
    max_row_margin: float | None = None,
    max_column_margin: float | None = None,
) -> np.ndarray:
    """Return a low-weight activity adjustment for additive tie-breaking.

    Activity is deliberately treated as a *centered* weak tie-breaker rather than
    as a primary evidence source. The neutral activity cost is subtracted before
    scaling, so good activity agreement can slightly lower an otherwise spatially
    plausible edge and poor agreement can slightly raise it. Missing activity
    evidence contributes exactly zero when an availability component is present.

    Optional row/column margin gates restrict the adjustment to candidates that
    are already competitive under the spatial/ROI cost matrix. This keeps
    activity from rescuing geometrically implausible edges while still allowing it
    to break local ties among near-equal candidates.
    """

    weight = float(weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    neutral_cost = float(neutral_cost)
    if not np.isfinite(neutral_cost):
        raise ValueError("neutral_cost must be finite")

    values = _pairwise_component(pairwise_components, component_name)
    if values.size == 0 or weight == 0.0:
        return np.zeros_like(values, dtype=float)

    finite_values = np.nan_to_num(
        values,
        nan=neutral_cost,
        posinf=1.0e6,
        neginf=0.0,
    )
    adjustment = finite_values - neutral_cost
    availability = _availability_mask(
        pairwise_components,
        availability_component=availability_component,
        shape=values.shape,
    )
    adjustment = np.where(availability, adjustment, 0.0)

    if max_row_margin is not None or max_column_margin is not None:
        if base_cost_matrix is None:
            raise ValueError(
                "base_cost_matrix is required when row/column margin gating is requested"
            )
        adjustment = np.where(
            _competitive_candidate_mask(
                base_cost_matrix,
                shape=values.shape,
                max_row_margin=max_row_margin,
                max_column_margin=max_column_margin,
            ),
            adjustment,
            0.0,
        )

    return weight * adjustment


def _pairwise_component(
    pairwise_components: Mapping[str, Any], component_name: str
) -> np.ndarray:
    if component_name not in pairwise_components:
        raise KeyError(f"Pairwise components do not contain {component_name!r}")
    values = np.asarray(pairwise_components[component_name], dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"Pairwise component {component_name!r} must be two-dimensional"
        )
    return values


def _availability_mask(
    pairwise_components: Mapping[str, Any],
    *,
    availability_component: str | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if availability_component is None:
        return np.ones(shape, dtype=bool)
    if availability_component not in pairwise_components:
        return np.ones(shape, dtype=bool)
    availability = np.asarray(pairwise_components[availability_component], dtype=float)
    if availability.shape != shape:
        raise ValueError(
            f"Pairwise component {availability_component!r} must have shape {shape}"
        )
    return np.isfinite(availability) & (availability > 0.0)


def _competitive_candidate_mask(
    base_cost_matrix: Any,
    *,
    shape: tuple[int, int],
    max_row_margin: float | None,
    max_column_margin: float | None,
) -> np.ndarray:
    base_costs = np.asarray(base_cost_matrix, dtype=float)
    if base_costs.shape != shape:
        raise ValueError(f"base_cost_matrix must have shape {shape}")
    if base_costs.size == 0:
        return np.ones(shape, dtype=bool)

    finite_base = np.nan_to_num(base_costs, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
    mask = np.ones(shape, dtype=bool)
    if max_row_margin is not None:
        max_row_margin = float(max_row_margin)
        if max_row_margin < 0.0:
            raise ValueError("max_row_margin must be non-negative")
        row_best = np.min(finite_base, axis=1, keepdims=True)
        mask &= (finite_base - row_best) <= max_row_margin
    if max_column_margin is not None:
        max_column_margin = float(max_column_margin)
        if max_column_margin < 0.0:
            raise ValueError("max_column_margin must be non-negative")
        column_best = np.min(finite_base, axis=0, keepdims=True)
        mask &= (finite_base - column_best) <= max_column_margin
    return mask
