"""Scaffolding for iterative joint registration/assignment refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

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
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0.0 <= self.high_confidence_quantile <= 1.0:
            raise ValueError("high_confidence_quantile must lie in [0, 1]")
        if self.min_anchor_edges < 1:
            raise ValueError("min_anchor_edges must be positive")
        if self.cost_relief < 0.0:
            raise ValueError("cost_relief must be non-negative")


@dataclass(frozen=True)
class JointRegistrationAssignmentConfig:
    """Controls for selecting high-confidence assignment anchors."""

    min_anchor_probability: float = 0.90
    min_anchor_margin: float = 0.20
    min_anchors: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_anchor_probability <= 1.0:
            raise ValueError("min_anchor_probability must lie in [0, 1]")
        if self.min_anchor_margin < 0.0:
            raise ValueError("min_anchor_margin must be non-negative")
        if self.min_anchors < 1:
            raise ValueError("min_anchors must be positive")


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
