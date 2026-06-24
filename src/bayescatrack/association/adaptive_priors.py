"""Adaptive edge-prior utilities for global multi-session assignment.

PyRecEst's current path-cover solver consumes scalar start/end/gap priors plus
pairwise edge-cost matrices.  This module provides a light-weight bridge toward
learned or ROI-conditioned priors without changing the solver API: ROI quality,
cell probability, FOV-border and session-gap priors are folded into the pairwise
edge matrices before the solver is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.advanced_roi_components import mask_shape_descriptors
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.reference import Track2pReference

SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class AdaptiveEdgePriorConfig:
    """Weights for ROI-conditioned edge-prior adjustments."""

    session_gap_weight: float = 0.0
    border_proximity_weight: float = 0.0
    low_cell_probability_weight: float = 0.0
    mask_fragility_weight: float = 0.0
    learned_gap_costs: Mapping[int, float] | None = None
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        for name in (
            "session_gap_weight",
            "border_proximity_weight",
            "low_cell_probability_weight",
            "mask_fragility_weight",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_float(getattr(self, name), name=name),
            )
        large_cost = _finite_positive_float(self.large_cost, name="large_cost")
        object.__setattr__(self, "large_cost", large_cost)
        if self.learned_gap_costs is not None:
            object.__setattr__(
                self,
                "learned_gap_costs",
                _validated_learned_gap_costs(self.learned_gap_costs),
            )

    @property
    def enabled(self) -> bool:
        return (
            self.session_gap_weight > 0.0
            or self.border_proximity_weight > 0.0
            or self.low_cell_probability_weight > 0.0
            or self.mask_fragility_weight > 0.0
            or bool(self.learned_gap_costs)
        )


def apply_adaptive_edge_priors(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    sessions: Sequence[Track2pSession],
    *,
    config: AdaptiveEdgePriorConfig | Mapping[str, Any] | None = None,
) -> dict[SessionEdge, np.ndarray]:
    """Return pairwise costs with ROI-conditioned prior adjustments added."""

    resolved = _coerce_config(config)
    sessions = tuple(sessions)
    copied = {
        (int(edge[0]), int(edge[1])): np.asarray(matrix, dtype=float).copy()
        for edge, matrix in pairwise_costs.items()
    }
    if not resolved.enabled:
        return copied

    quality = tuple(_session_roi_quality(session) for session in sessions)
    adjusted: dict[SessionEdge, np.ndarray] = {}
    for edge, matrix in copied.items():
        source, target = edge
        if source < 0 or target <= source or target >= len(sessions):
            raise ValueError(f"Invalid session edge {edge!r}")
        if matrix.shape != (
            sessions[source].plane_data.n_rois,
            sessions[target].plane_data.n_rois,
        ):
            raise ValueError(
                f"Pairwise cost matrix for edge {edge!r} has shape {matrix.shape}, "
                "which does not match the loaded session ROI counts"
            )
        edge_cost = matrix.copy()
        admissible = np.isfinite(edge_cost) & (edge_cost < resolved.large_cost)
        gap = int(target - source)
        if resolved.session_gap_weight > 0.0:
            edge_cost[admissible] += resolved.session_gap_weight * max(gap - 1, 0)
        if resolved.learned_gap_costs:
            edge_cost[admissible] += float(resolved.learned_gap_costs.get(gap, 0.0))
        if resolved.border_proximity_weight > 0.0:
            edge_cost += resolved.border_proximity_weight * _pairwise_mean_quality(
                quality[source]["border_proximity"],
                quality[target]["border_proximity"],
            )
        if resolved.low_cell_probability_weight > 0.0:
            edge_cost += resolved.low_cell_probability_weight * _pairwise_mean_quality(
                quality[source]["low_cell_probability"],
                quality[target]["low_cell_probability"],
            )
        if resolved.mask_fragility_weight > 0.0:
            edge_cost += resolved.mask_fragility_weight * _pairwise_mean_quality(
                quality[source]["mask_fragility"],
                quality[target]["mask_fragility"],
            )
        edge_cost[~admissible] = matrix[~admissible]
        adjusted[edge] = _finite_costs(edge_cost, large_cost=resolved.large_cost)
    return adjusted


def fit_gap_costs_from_reference(
    reference: Track2pReference,
    *,
    max_gap: int = 3,
    curated_only: bool = False,
    smoothing: float = 1.0,
) -> dict[int, float]:
    """Estimate simple positive-link gap costs from a reference matrix.

    The returned value is ``-log(P(link across gap))`` with Laplace smoothing.
    It is intentionally simple and meant as an interpretable prior, not a final
    discriminative model.
    """

    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    matrix = reference.filtered_indices(curated_only=curated_only)
    present = np.vectorize(lambda value: value is not None, otypes=[bool])(matrix)
    costs: dict[int, float] = {}
    for gap in range(1, int(max_gap) + 1):
        opportunities = 0
        positives = 0
        for source in range(reference.n_sessions - gap):
            target = source + gap
            source_present = present[:, source]
            target_present = present[:, target]
            opportunities += int(np.count_nonzero(source_present))
            positives += int(np.count_nonzero(source_present & target_present))
        probability = (positives + smoothing) / max(
            opportunities + 2.0 * smoothing, 1.0
        )
        costs[gap] = float(-np.log(np.clip(probability, 1.0e-12, 1.0)))
    return costs


def session_roi_quality(session: Track2pSession) -> dict[str, np.ndarray]:
    """Return ROI-quality vectors used by adaptive edge priors."""

    return _session_roi_quality(session)


def _session_roi_quality(session: Track2pSession) -> dict[str, np.ndarray]:
    plane = session.plane_data
    descriptors = mask_shape_descriptors(plane.roi_masks)
    n_rois = int(plane.n_rois)
    if plane.cell_probabilities is None:
        low_cell_probability = np.zeros((n_rois,), dtype=float)
    else:
        probabilities = np.clip(
            np.asarray(plane.cell_probabilities, dtype=float).reshape(n_rois),
            1.0e-6,
            1.0,
        )
        low_cell_probability = 1.0 - probabilities
    mask_fragility = np.maximum(
        descriptors["empty_mask"].astype(float),
        1.0 - np.clip(descriptors["compactness"], 0.0, 1.0),
    )
    return {
        "border_proximity": np.asarray(descriptors["border_proximity"], dtype=float),
        "low_cell_probability": np.asarray(low_cell_probability, dtype=float),
        "mask_fragility": np.asarray(mask_fragility, dtype=float),
    }


def _pairwise_mean_quality(
    reference: np.ndarray, measurement: np.ndarray
) -> np.ndarray:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    measurement = np.asarray(measurement, dtype=float).reshape(-1)
    return 0.5 * (reference[:, None] + measurement[None, :])


def _finite_costs(costs: np.ndarray, *, large_cost: float) -> np.ndarray:
    result = np.asarray(costs, dtype=float).copy()
    invalid = ~np.isfinite(result)
    result[invalid] = large_cost
    result[result < 0.0] = 0.0
    return result


def _validated_learned_gap_costs(
    learned_gap_costs: Mapping[int, float],
) -> dict[int, float]:
    coerced: dict[int, float] = {}
    for raw_gap, raw_cost in learned_gap_costs.items():
        gap = _positive_int(raw_gap, name="learned_gap_costs key")
        cost = _finite_nonnegative_float(raw_cost, name="learned_gap_costs value")
        coerced[gap] = cost
    return coerced


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite value")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return numeric


def _coerce_config(
    config: AdaptiveEdgePriorConfig | Mapping[str, Any] | None,
) -> AdaptiveEdgePriorConfig:
    if config is None:
        return AdaptiveEdgePriorConfig()
    if isinstance(config, AdaptiveEdgePriorConfig):
        return config
    return AdaptiveEdgePriorConfig(**dict(config))


__all__ = [
    "AdaptiveEdgePriorConfig",
    "apply_adaptive_edge_priors",
    "fit_gap_costs_from_reference",
    "session_roi_quality",
]
