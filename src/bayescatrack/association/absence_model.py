"""Observation-absence likelihoods for missed, split, or out-of-FOV cells."""

from __future__ import annotations

import operator
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
            value = _validated_non_negative_finite_float(name, getattr(self, name))
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
    n_rois = _validated_roi_count(plane, "plane")
    costs = np.full((n_rois,), float(cfg.base_absence_cost), dtype=float)

    cell_probabilities = getattr(plane, "cell_probabilities", None)
    if cell_probabilities is not None:
        probabilities = np.asarray(cell_probabilities, dtype=float).reshape(-1)
        if probabilities.shape == (n_rois,):
            valid_probabilities = np.isfinite(probabilities)
            if np.any(valid_probabilities):
                probs = np.clip(probabilities[valid_probabilities], 0.0, 1.0)
                costs[valid_probabilities] -= cfg.low_cell_probability_discount * (
                    1.0 - probs
                )

    if registered_empty_mask is not None:
        empty = np.asarray(registered_empty_mask, dtype=bool).reshape(-1)
        if empty.shape == (n_rois,):
            costs[empty] -= cfg.empty_registered_mask_discount

    if local_density is not None:
        density = np.asarray(local_density, dtype=float).reshape(-1)
        if density.shape == (n_rois,) and density.size:
            finite_density = density[np.isfinite(density)]
            if finite_density.size:
                scale = float(np.percentile(finite_density, 90.0))
            else:
                scale = 1.0
            if not np.isfinite(scale) or scale <= 1.0e-12:
                scale = 1.0
            sanitized_density = np.nan_to_num(
                density,
                nan=0.0,
                posinf=scale,
                neginf=0.0,
            )
            costs -= cfg.high_local_density_discount * np.clip(
                sanitized_density / scale, 0.0, 1.0
            )

    if (
        getattr(plane, "traces", None) is None
        and getattr(plane, "spike_traces", None) is None
    ):
        costs -= cfg.trace_missing_discount

    return np.maximum(costs, float(cfg.min_cost))


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
    n_ref = _validated_roi_count(reference_plane, "reference_plane")
    n_meas = _validated_roi_count(measurement_plane, "measurement_plane")
    if reference_absence_costs is None:
        ref_cost = absence_cost_vector(
            reference_plane,
            local_density=reference_local_density,
            config=cfg,
        )
    else:
        ref_cost = _validated_absence_cost_vector(
            "reference_absence_costs",
            reference_absence_costs,
            n_ref,
        )
    if measurement_absence_costs is None:
        meas_cost = absence_cost_vector(
            measurement_plane,
            registered_empty_mask=registered_empty_mask,
            local_density=measurement_local_density,
            config=cfg,
        )
    else:
        meas_cost = _validated_absence_cost_vector(
            "measurement_absence_costs",
            measurement_absence_costs,
            n_meas,
        )
    if ref_cost.shape != (n_ref,) or meas_cost.shape != (n_meas,):
        raise ValueError("absence cost vectors must match plane ROI counts")
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
    gap_penalties = gap_penalty_matrix(
        reference_plane,
        measurement_plane,
        session_gap=session_gap,
        registered_empty_mask=registered_empty_mask,
        reference_local_density=reference_local_density,
        measurement_local_density=measurement_local_density,
        config=cfg,
    )
    if costs.shape != gap_penalties.shape:
        raise ValueError(
            "cost_matrix shape must match plane ROI counts: "
            f"expected {gap_penalties.shape}, got {costs.shape}"
        )
    return costs + gap_penalties


def absence_summary(plane: Any, *, costs: Any | None = None) -> dict[str, float | int]:
    """Return scalar diagnostics for absence modeling."""

    if costs is None:
        cost_values = absence_cost_vector(plane)
    else:
        cost_values = np.asarray(costs, dtype=float).reshape(-1)
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


def _validated_roi_count(plane: Any, plane_name: str) -> int:
    message = f"{plane_name}.n_rois must be a finite non-negative integer"
    raw_count = getattr(plane, "n_rois", 0)
    if isinstance(raw_count, (bool, np.bool_)):
        raise ValueError(message)

    try:
        count = int(operator.index(raw_count))
    except TypeError:
        try:
            numeric_count = float(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(numeric_count) or not numeric_count.is_integer():
            raise ValueError(message)
        count = int(numeric_count)

    if count < 0:
        raise ValueError(message)
    return count


def _validated_non_negative_finite_float(name: str, raw_value: Any) -> float:
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validated_absence_cost_vector(name: str, raw_values: Any, n_rois: int) -> np.ndarray:
    values = np.asarray(raw_values, dtype=float).reshape(-1)
    if values.shape != (n_rois,):
        raise ValueError("absence cost vectors must match plane ROI counts")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


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


__all__ = (
    "AbsenceModelConfig",
    "absence_model_config_from_mapping",
    "absence_cost_vector",
    "gap_penalty_matrix",
    "apply_absence_adjustment",
    "absence_summary",
)
