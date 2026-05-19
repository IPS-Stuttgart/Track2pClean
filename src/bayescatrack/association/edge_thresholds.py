"""Edge-specific cost-threshold policies for multi-session assignment.

The PyRecEst path-cover solver accepts one global cost threshold.  For the
Track2p benchmark this is often too coarse: each session edge can have a very
different registered-overlap distribution.  This module provides deterministic
pre-solver filters that mark edge-specific rejected links as ``np.inf`` while
leaving the solver objective itself unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np

SessionEdge = tuple[int, int]
EdgeThresholdPolicy = Literal["none", "otsu", "manual-oracle"]


def compute_otsu_edge_cost_thresholds(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    bins: int = 256,
    max_cost: float | None = None,
) -> dict[SessionEdge, float | None]:
    """Return one unsupervised Otsu threshold per session edge.

    Costs are lower-is-better.  The returned threshold is therefore a maximum
    accepted cost.  ``max_cost`` is an optional diagnostic guard that excludes
    intentionally huge finite sentinel costs, e.g. empty-registered-ROI
    penalties, from the Otsu histogram.
    """

    return {
        edge: otsu_cost_threshold(cost_matrix, bins=bins, max_cost=max_cost)
        for edge, cost_matrix in pairwise_costs.items()
    }


def compute_manual_oracle_edge_cost_thresholds(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    true_match_masks: Mapping[SessionEdge, np.ndarray],
) -> dict[SessionEdge, float | None]:
    """Return F1-optimal edge thresholds from binary ground-truth masks.

    This is a diagnostic oracle.  It should only be used to quantify how much
    of the benchmark gap comes from edge-threshold calibration, not as a final
    paper-facing unsupervised method.
    """

    thresholds: dict[SessionEdge, float | None] = {}
    for edge, cost_matrix in pairwise_costs.items():
        true_mask = true_match_masks.get(edge)
        if true_mask is None:
            true_mask = np.zeros_like(cost_matrix, dtype=bool)
        thresholds[edge] = manual_oracle_cost_threshold(cost_matrix, true_mask)
    return thresholds


def apply_edge_cost_thresholds(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    thresholds: Mapping[SessionEdge, float | None],
    *,
    rejected_cost: float = np.inf,
) -> dict[SessionEdge, np.ndarray]:
    """Reject links above each edge's threshold by replacing them with inf."""

    filtered: dict[SessionEdge, np.ndarray] = {}
    for edge, cost_matrix in pairwise_costs.items():
        cost_matrix = np.asarray(cost_matrix, dtype=float)
        threshold = thresholds.get(edge)
        filtered_matrix = cost_matrix.copy()
        if threshold is not None:
            threshold = float(threshold)
            if np.isneginf(threshold):
                finite_mask = np.isfinite(filtered_matrix)
                filtered_matrix[finite_mask] = rejected_cost
            else:
                filtered_matrix[filtered_matrix > threshold] = rejected_cost
        filtered[edge] = filtered_matrix
    return filtered


def otsu_cost_threshold(
    cost_matrix: np.ndarray,
    *,
    bins: int = 256,
    max_cost: float | None = None,
) -> float | None:
    """Compute a lower-is-better Otsu threshold for one cost matrix."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = _finite_cost_values(cost_matrix, max_cost=max_cost)
    if values.size == 0:
        return None
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if min_value == max_value:
        return min_value

    counts, bin_edges = np.histogram(
        values, bins=int(bins), range=(min_value, max_value)
    )
    total = float(np.sum(counts))
    if total <= 0.0:
        return None

    probabilities = counts.astype(float) / total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cumulative_weight = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * bin_centers)
    total_mean = float(cumulative_mean[-1])

    denominators = cumulative_weight * (1.0 - cumulative_weight)
    between_class_variance = np.full_like(bin_centers, -np.inf, dtype=float)
    valid = denominators > 0.0
    between_class_variance[valid] = (
        (total_mean * cumulative_weight[valid] - cumulative_mean[valid]) ** 2
        / denominators[valid]
    )
    if not np.any(np.isfinite(between_class_variance)):
        return max_value

    best_bin = int(np.nanargmax(between_class_variance))
    return float(bin_edges[best_bin + 1])


def manual_oracle_cost_threshold(
    cost_matrix: np.ndarray, true_match_mask: np.ndarray
) -> float | None:
    """Return the strictest cost threshold that maximizes pairwise F1."""

    costs = np.asarray(cost_matrix, dtype=float)
    labels = np.asarray(true_match_mask, dtype=bool)
    if labels.shape != costs.shape:
        raise ValueError("true_match_mask must have the same shape as cost_matrix")

    finite_mask = np.isfinite(costs)
    if not np.any(finite_mask):
        return None

    finite_costs = costs[finite_mask]
    finite_labels = labels[finite_mask]
    positives = int(np.count_nonzero(finite_labels))
    if positives == 0:
        return _reject_all_threshold(finite_costs)

    order = np.argsort(finite_costs, kind="mergesort")
    sorted_costs = finite_costs[order]
    sorted_labels = finite_labels[order]

    group_ends = np.flatnonzero(np.diff(sorted_costs))
    group_ends = np.concatenate([group_ends, np.asarray([sorted_costs.size - 1])])
    cumulative_true = np.cumsum(sorted_labels, dtype=int)

    accepted = group_ends + 1
    true_positives = cumulative_true[group_ends]
    false_positives = accepted - true_positives
    false_negatives = positives - true_positives
    denominators = 2 * true_positives + false_positives + false_negatives
    f1_scores = np.divide(
        2 * true_positives,
        denominators,
        out=np.zeros_like(denominators, dtype=float),
        where=denominators > 0,
    )

    # ``argmax`` returns the first maximum, i.e. the strictest threshold among
    # equally good F1 values because groups are sorted by increasing cost.
    best_group = int(np.argmax(f1_scores))
    return float(sorted_costs[group_ends[best_group]])


def _finite_cost_values(
    cost_matrix: np.ndarray, *, max_cost: float | None
) -> np.ndarray:
    values = np.asarray(cost_matrix, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if max_cost is not None:
        finite &= values <= float(max_cost)
    return values[finite]


def _reject_all_threshold(finite_costs: np.ndarray) -> float:
    min_cost = float(np.min(np.asarray(finite_costs, dtype=float)))
    return float(np.nextafter(min_cost, -np.inf))
