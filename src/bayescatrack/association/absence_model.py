"""Observation-absence likelihoods for missed, split, or out-of-FOV cells."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
        for name in (
            "base_absence_cost",
            "out_of_fov_discount",
            "low_cell_probability_discount",
            "empty_registered_mask_discount",
            "high_local_density_discount",
            "trace_missing_discount",
            "min_cost",
        ):
            value = _finite_nonnegative_config_value(getattr(self, name), name=name)
            object.__setattr__(self, name, value)


def absence_model_config_from_mapping(
    value: AbsenceModelConfig | Mapping[str, Any] | None,
) -> AbsenceModelConfig | None:
    """Normalize optional absence-model config values."""

    if value is None:
        return None
    if isinstance(value, AbsenceModelConfig):
        return value
    return AbsenceModelConfig(**dict(value))


def absence_cost_vector(
    plane: Any,
    *,
    registered_empty_mask: Any | None = None,
    local_density: Any | None = None,
    config: AbsenceModelConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return per-ROI costs for allowing an observation gap/absence."""

    cfg = absence_model_config_from_mapping(config) or AbsenceModelConfig()
    n_rois = int(getattr(plane, "n_rois", 0))
    costs = np.full((n_rois,), float(cfg.base_absence_cost), dtype=float)

    cell_probabilities = getattr(plane, "cell_probabilities", None)
    if cell_probabilities is not None:
        probs = _sanitized_probability_vector(cell_probabilities, n_rois=n_rois)
        if probs is not None:
            costs -= cfg.low_cell_probability_discount * (1.0 - probs)

    if registered_empty_mask is not None:
        empty = np.asarray(registered_empty_mask, dtype=bool).reshape(-1)
        if empty.shape == (n_rois,):
            costs[empty] -= cfg.empty_registered_mask_discount

    if local_density is not None:
        density = _sanitized_local_density_vector(local_density, n_rois=n_rois)
        if density is not None and density.size:
            scale = float(np.percentile(density, 90.0))
            if not np.isfinite(scale) or scale <= 1.0e-12:
                scale = 1.0
            costs -= cfg.high_local_density_discount * np.clip(
                density / scale, 0.0, 1.0
            )

    if (
        getattr(plane, "traces", None) is None
        and getattr(plane, "spike_traces", None) is None
    ):
        costs -= cfg.trace_missing_discount

    costs = np.maximum(costs, float(cfg.min_cost))
    if not np.all(np.isfinite(costs)):
        raise ValueError("absence costs must contain only finite values")
    return costs


def gap_penalty_matrix(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    session_gap: int | float = 1.0,
    reference_absence_costs: Any | None = None,
    measurement_absence_costs: Any | None = None,
    registered_empty_mask: Any | None = None,
    reference_local_density: Any | None = None,
    measurement_local_density: Any | None = None,
    config: AbsenceModelConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return pairwise gap penalties that account for observation absence cues."""

    cfg = absence_model_config_from_mapping(config) or AbsenceModelConfig()
    n_ref = int(getattr(reference_plane, "n_rois", 0))
    n_meas = int(getattr(measurement_plane, "n_rois", 0))
    if reference_absence_costs is None:
        ref_cost = absence_cost_vector(
            reference_plane,
            local_density=reference_local_density,
            config=cfg,
        )
    else:
        ref_cost = _normalize_absence_cost_vector(
            reference_absence_costs,
            n_rois=n_ref,
            name="reference_absence_costs",
        )
    if measurement_absence_costs is None:
        meas_cost = absence_cost_vector(
            measurement_plane,
            registered_empty_mask=registered_empty_mask,
            local_density=measurement_local_density,
            config=cfg,
        )
    else:
        meas_cost = _normalize_absence_cost_vector(
            measurement_absence_costs,
            n_rois=n_meas,
            name="measurement_absence_costs",
        )
    gap = _validated_session_gap_offset(session_gap)
    return gap * 0.5 * (ref_cost[:, None] + meas_cost[None, :])


def apply_absence_adjustment(
    cost_matrix: Any,
    reference_plane: Any,
    measurement_plane: Any,
    *,
    session_gap: int | float = 1.0,
    registered_empty_mask: Any | None = None,
    reference_local_density: Any | None = None,
    measurement_local_density: Any | None = None,
    config: AbsenceModelConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Add absence-aware gap penalties to a cost matrix."""

    costs = np.asarray(cost_matrix, dtype=float)
    cfg = absence_model_config_from_mapping(config) or AbsenceModelConfig()
    adjusted = costs + gap_penalty_matrix(
        reference_plane,
        measurement_plane,
        session_gap=session_gap,
        registered_empty_mask=registered_empty_mask,
        reference_local_density=reference_local_density,
        measurement_local_density=measurement_local_density,
        config=cfg,
    )
    if not np.all(np.isfinite(adjusted)):
        raise ValueError("absence-adjusted cost matrix must contain only finite values")
    return adjusted


def absence_summary(plane: Any, *, costs: Any | None = None) -> dict[str, float | int]:
    """Return scalar diagnostics for absence modeling."""

    if costs is None:
        cost_values = absence_cost_vector(plane)
    else:
        cost_values = np.asarray(costs, dtype=float).reshape(-1)
        if not np.all(np.isfinite(cost_values)):
            raise ValueError("costs must contain only finite values")
    return {
        "n_rois": int(cost_values.size),
        "mean_absence_cost": (
            float(np.mean(cost_values)) if cost_values.size else float("nan")
        ),
        "median_absence_cost": (
            float(np.median(cost_values)) if cost_values.size else float("nan")
        ),
        "min_absence_cost": (
            float(np.min(cost_values)) if cost_values.size else float("nan")
        ),
        "max_absence_cost": (
            float(np.max(cost_values)) if cost_values.size else float("nan")
        ),
    }


def _finite_nonnegative_config_value(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _sanitized_probability_vector(value: Any, *, n_rois: int) -> np.ndarray | None:
    probabilities = np.asarray(value, dtype=float).reshape(-1)
    if probabilities.shape != (n_rois,):
        return None
    probabilities = np.nan_to_num(
        probabilities,
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(probabilities, 0.0, 1.0)


def _sanitized_local_density_vector(value: Any, *, n_rois: int) -> np.ndarray | None:
    density = np.asarray(value, dtype=float).reshape(-1)
    if density.shape != (n_rois,):
        return None
    return np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_absence_cost_vector(value: Any, *, n_rois: int, name: str) -> np.ndarray:
    costs = np.asarray(value, dtype=float).reshape(-1)
    if costs.shape != (n_rois,):
        raise ValueError("absence cost vectors must match plane ROI counts")
    if not np.all(np.isfinite(costs)):
        raise ValueError(f"{name} must contain only finite values")
    return costs


def _validated_session_gap_offset(session_gap: int | float) -> float:
    if isinstance(session_gap, (bool, np.bool_)):
        raise ValueError(
            "session_gap must be a finite value greater than or equal to 1"
        )
    gap = float(session_gap)
    if not np.isfinite(gap) or gap < 1.0:
        raise ValueError(
            "session_gap must be a finite value greater than or equal to 1"
        )
    return gap - 1.0


__all__ = (
    "AbsenceModelConfig",
    "absence_model_config_from_mapping",
    "absence_cost_vector",
    "gap_penalty_matrix",
    "apply_absence_adjustment",
    "absence_summary",
)
