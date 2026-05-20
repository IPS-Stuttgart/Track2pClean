"""Observation-absence likelihoods for missed, split, or out-of-FOV cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class AbsenceModelConfig:
    """Weights for converting observability cues into gap/death penalties."""

    base_absence_cost: float = 1.0
    out_of_fov_discount: float = 0.75
    low_cell_probability_discount: float = 0.50
    empty_registered_mask_discount: float = 0.75
    high_local_density_discount: float = 0.25
    trace_missing_discount: float = 0.10
    min_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.base_absence_cost < 0.0:
            raise ValueError("base_absence_cost must be non-negative")
        if self.min_cost < 0.0:
            raise ValueError("min_cost must be non-negative")


def absence_cost_vector(
    plane: Any,
    *,
    registered_empty_mask: Any | None = None,
    local_density: Any | None = None,
    config: AbsenceModelConfig | None = None,
) -> np.ndarray:
    """Return per-ROI costs for allowing an observation gap/absence."""

    cfg = config or AbsenceModelConfig()
    n_rois = int(getattr(plane, "n_rois", 0))
    costs = np.full((n_rois,), float(cfg.base_absence_cost), dtype=float)

    cell_probabilities = getattr(plane, "cell_probabilities", None)
    if cell_probabilities is not None:
        probs = np.clip(np.asarray(cell_probabilities, dtype=float).reshape(-1), 0.0, 1.0)
        if probs.shape == (n_rois,):
            costs -= cfg.low_cell_probability_discount * (1.0 - probs)

    if registered_empty_mask is not None:
        empty = np.asarray(registered_empty_mask, dtype=bool).reshape(-1)
        if empty.shape == (n_rois,):
            costs[empty] -= cfg.empty_registered_mask_discount

    if local_density is not None:
        density = np.asarray(local_density, dtype=float).reshape(-1)
        if density.shape == (n_rois,) and density.size:
            scale = float(np.nanpercentile(density, 90.0))
            if not np.isfinite(scale) or scale <= 1.0e-12:
                scale = 1.0
            costs -= cfg.high_local_density_discount * np.clip(density / scale, 0.0, 1.0)

    if getattr(plane, "traces", None) is None and getattr(plane, "spike_traces", None) is None:
        costs -= cfg.trace_missing_discount

    return np.maximum(costs, float(cfg.min_cost))


def gap_penalty_matrix(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    session_gap: int | float = 1.0,
    reference_absence_costs: Any | None = None,
    measurement_absence_costs: Any | None = None,
    config: AbsenceModelConfig | None = None,
) -> np.ndarray:
    """Return pairwise gap penalties that account for observation absence cues."""

    cfg = config or AbsenceModelConfig()
    n_ref = int(getattr(reference_plane, "n_rois", 0))
    n_meas = int(getattr(measurement_plane, "n_rois", 0))
    if reference_absence_costs is None:
        ref_cost = absence_cost_vector(reference_plane, config=cfg)
    else:
        ref_cost = np.asarray(reference_absence_costs, dtype=float).reshape(-1)
    if measurement_absence_costs is None:
        meas_cost = absence_cost_vector(measurement_plane, config=cfg)
    else:
        meas_cost = np.asarray(measurement_absence_costs, dtype=float).reshape(-1)
    if ref_cost.shape != (n_ref,) or meas_cost.shape != (n_meas,):
        raise ValueError("absence cost vectors must match plane ROI counts")
    gap = max(float(session_gap) - 1.0, 0.0)
    return gap * 0.5 * (ref_cost[:, None] + meas_cost[None, :])


def apply_absence_adjustment(cost_matrix: Any, reference_plane: Any, measurement_plane: Any, *, session_gap: int | float = 1.0, config: AbsenceModelConfig | None = None) -> np.ndarray:
    """Add absence-aware gap penalties to a cost matrix."""

    costs = np.asarray(cost_matrix, dtype=float)
    return costs + gap_penalty_matrix(
        reference_plane,
        measurement_plane,
        session_gap=session_gap,
        config=config,
    )


def absence_summary(plane: Any, *, costs: Any | None = None) -> dict[str, float | int]:
    """Return scalar diagnostics for absence modeling."""

    if costs is None:
        cost_values = absence_cost_vector(plane)
    else:
        cost_values = np.asarray(costs, dtype=float).reshape(-1)
    return {
        "n_rois": int(cost_values.size),
        "mean_absence_cost": float(np.mean(cost_values)) if cost_values.size else float("nan"),
        "median_absence_cost": float(np.median(cost_values)) if cost_values.size else float("nan"),
        "min_absence_cost": float(np.min(cost_values)) if cost_values.size else float("nan"),
        "max_absence_cost": float(np.max(cost_values)) if cost_values.size else float("nan"),
    }
