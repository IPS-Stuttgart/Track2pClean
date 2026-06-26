"""Scaffolding for iterative joint registration/assignment refinement."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JointRefinementConfig:
    """Controls for iterative registration/assignment refinement."""

    max_iterations: int = 3
    high_confidence_quantile: float = 0.10
    min_anchor_edges: int = 8
    cost_relief: float = 0.25
    convergence_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_iterations",
            _validate_positive_int(self.max_iterations, "max_iterations"),
        )
        object.__setattr__(
            self,
            "high_confidence_quantile",
            _validate_probability(
                self.high_confidence_quantile,
                "high_confidence_quantile",
            ),
        )
        object.__setattr__(
            self,
            "min_anchor_edges",
            _validate_positive_int(self.min_anchor_edges, "min_anchor_edges"),
        )
        object.__setattr__(
            self,
            "cost_relief",
            _validate_finite_nonnegative(self.cost_relief, "cost_relief"),
        )
        object.__setattr__(
            self,
            "convergence_tolerance",
            _validate_finite_nonnegative(
                self.convergence_tolerance,
                "convergence_tolerance",
            ),
        )


@dataclass(frozen=True)
class JointRegistrationAssignmentConfig:
    """Controls for selecting high-confidence assignment anchors."""

    min_anchor_probability: float = 0.90
    min_anchor_margin: float = 0.20
    min_anchors: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_anchor_probability",
            _validate_probability(
                self.min_anchor_probability,
                "min_anchor_probability",
            ),
        )
        object.__setattr__(
            self,
            "min_anchor_margin",
            _validate_finite_nonnegative(
                self.min_anchor_margin,
                "min_anchor_margin",
            ),
        )
        object.__setattr__(
            self,
            "min_anchors",
            _validate_positive_int(self.min_anchors, "min_anchors"),
        )


@dataclass(frozen=True)
class JointRefinementState:
    """One iteration of joint refinement."""

    iteration: int
    registered_plane: Any
    cost_matrix: np.ndarray
    anchor_edges: tuple[tuple[int, int], ...]
    mean_anchor_cost: float


def high_confidence_anchor_pairs(
    probabilities: Any,
    *,
    config: JointRegistrationAssignmentConfig | None = None,
) -> np.ndarray:
    """Return row-best probability anchors with enough confidence and margin."""

    cfg = config or JointRegistrationAssignmentConfig()
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if probs.shape[1] == 0:
        return np.zeros((0, 2), dtype=int)
    anchors: list[tuple[int, int]] = []
    for row_index, row in enumerate(probs):
        finite_row = np.where(np.isfinite(row), row, -np.inf)
        col_index = int(np.argmax(finite_row))
        best = float(finite_row[col_index])
        if probs.shape[1] > 1:
            runner_up = float(np.max(np.delete(finite_row, col_index)))
        else:
            runner_up = 0.0
        if (
            best >= cfg.min_anchor_probability
            and best - runner_up >= cfg.min_anchor_margin
        ):
            anchors.append((int(row_index), col_index))
    if len(anchors) < cfg.min_anchors:
        flat = np.flatnonzero(np.isfinite(probs.reshape(-1)))
        order = flat[np.argsort(probs.reshape(-1)[flat])[::-1]]
        for index in order:
            candidate = (int(index // probs.shape[1]), int(index % probs.shape[1]))
            if candidate not in anchors:
                anchors.append(candidate)
            if len(anchors) >= cfg.min_anchors:
                break
    return np.asarray(anchors, dtype=int).reshape(-1, 2)


def high_confidence_anchor_edges(
    cost_matrix: Any,
    *,
    quantile: float = 0.10,
    min_anchor_edges: int = 8,
) -> tuple[tuple[int, int], ...]:
    """Return mutual low-cost row/column anchors from a cost matrix."""

    quantile = _validate_probability(quantile, "quantile")
    min_anchor_edges = _validate_positive_int(min_anchor_edges, "min_anchor_edges")
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    finite = np.isfinite(costs)
    if not np.any(finite):
        return ()
    threshold = float(np.quantile(costs[finite], quantile))
    row_best = np.argmin(np.where(finite, costs, np.inf), axis=1)
    col_best = np.argmin(np.where(finite, costs, np.inf), axis=0)
    anchors: list[tuple[int, int]] = []
    for row_index, col_index in enumerate(row_best):
        if not finite[row_index, col_index] or costs[row_index, col_index] > threshold:
            continue
        if int(col_best[col_index]) == row_index:
            anchors.append((int(row_index), int(col_index)))
    if len(anchors) < min_anchor_edges:
        flat = np.flatnonzero(finite.reshape(-1))
        order = flat[np.argsort(costs.reshape(-1)[flat])[:min_anchor_edges]]
        anchors.extend(
            (int(index // costs.shape[1]), int(index % costs.shape[1]))
            for index in order
            if (int(index // costs.shape[1]), int(index % costs.shape[1]))
            not in anchors
        )
    return tuple(anchors[: max(min_anchor_edges, len(anchors))])


def anchor_relief_cost_matrix(
    cost_matrix: Any,
    anchor_edges: Sequence[tuple[int, int]],
    *,
    relief: float = 0.25,
) -> np.ndarray:
    """Return a copy of costs with a small relief around trusted anchors."""

    relief = _validate_finite_nonnegative(relief, "relief")
    costs = np.asarray(cost_matrix, dtype=float).copy()
    if relief <= 0.0 or not anchor_edges:
        return costs
    for row_index, col_index in anchor_edges:
        if 0 <= row_index < costs.shape[0] and 0 <= col_index < costs.shape[1]:
            costs[row_index, col_index] = max(0.0, costs[row_index, col_index] - relief)
    return costs


def joint_registration_assignment_loop(
    reference_plane: Any,
    moving_plane: Any,
    *,
    register_fn: Callable[[Any, Any, Sequence[tuple[int, int]] | None], Any],
    cost_fn: Callable[[Any, Any], np.ndarray],
    config: JointRefinementConfig | None = None,
) -> list[JointRefinementState]:
    """Iteratively refine registration using high-confidence assignment anchors.

    The function is intentionally callback-based because BayesCaTrack already has
    several registration backends.  ``register_fn`` receives the reference plane,
    moving plane, and optional anchor edges from the previous iteration, then
    returns a registered moving plane.  ``cost_fn`` computes the cost matrix for
    the reference and registered moving planes.
    """

    cfg = config or JointRefinementConfig()
    states: list[JointRefinementState] = []
    anchors: tuple[tuple[int, int], ...] | None = None
    previous_mean = np.inf
    for iteration in range(cfg.max_iterations):
        registered_plane = register_fn(reference_plane, moving_plane, anchors)
        cost_matrix = np.asarray(
            cost_fn(reference_plane, registered_plane), dtype=float
        )
        anchors = high_confidence_anchor_edges(
            cost_matrix,
            quantile=cfg.high_confidence_quantile,
            min_anchor_edges=cfg.min_anchor_edges,
        )
        if anchors:
            mean_anchor_cost = float(
                np.mean([cost_matrix[row, col] for row, col in anchors])
            )
        else:
            mean_anchor_cost = float("nan")
        states.append(
            JointRefinementState(
                iteration=int(iteration),
                registered_plane=registered_plane,
                cost_matrix=cost_matrix,
                anchor_edges=anchors,
                mean_anchor_cost=mean_anchor_cost,
            )
        )
        if (
            np.isfinite(mean_anchor_cost)
            and abs(previous_mean - mean_anchor_cost) <= cfg.convergence_tolerance
        ):
            break
        previous_mean = mean_anchor_cost
    return states


def apply_joint_anchor_relief_to_pairwise_costs(
    pairwise_costs: Mapping[tuple[int, int], np.ndarray],
    *,
    config: JointRefinementConfig | Mapping[str, Any] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Apply one assignment-anchor refinement pass to every pairwise edge matrix.

    This is the solver-side hook for joint registration/assignment experiments:
    when registration backends do not yet consume anchors directly, we still make
    high-confidence mutual anchors visible to the assignment objective as small,
    bounded edge relief.  It is opt-in and leaves registration unchanged.
    """

    cfg = (
        config
        if isinstance(config, JointRefinementConfig)
        else JointRefinementConfig(**dict(config or {}))
    )
    adjusted: dict[tuple[int, int], np.ndarray] = {}
    for edge, matrix in pairwise_costs.items():
        anchors = high_confidence_anchor_edges(
            matrix,
            quantile=cfg.high_confidence_quantile,
            min_anchor_edges=cfg.min_anchor_edges,
        )
        adjusted[(int(edge[0]), int(edge[1]))] = anchor_relief_cost_matrix(
            matrix, anchors, relief=cfg.cost_relief
        )
    return adjusted


def state_summary_rows(
    states: Sequence[JointRefinementState],
) -> list[dict[str, int | float]]:
    """Serialize joint-refinement iteration diagnostics."""

    return [
        {
            "iteration": state.iteration,
            "anchor_edges": len(state.anchor_edges),
            "mean_anchor_cost": state.mean_anchor_cost,
            "finite_costs": int(np.count_nonzero(np.isfinite(state.cost_matrix))),
        }
        for state in states
    ]


def _validate_probability(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a finite probability in [0, 1]")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite probability in [0, 1]") from exc
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability in [0, 1]")
    return normalized


def _validate_finite_nonnegative(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a finite non-negative value")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative value") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative value")
    return normalized


def _validate_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    normalized = int(normalized)
    if normalized < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized
