"""Strict input validation for global assignment edge metadata.

PyRecEst's global assignment solver consumes a mapping of pairwise cost matrices.
BayesCaTrack additionally stores ``session_edges`` metadata in ``GlobalAssignmentRun``
for diagnostics, exports, and downstream benchmark analysis. If that metadata
advertises edges not present in the actual cost mapping, later diagnostics can
report empty or shifted link-cost columns for edges that were never solved. This
module installs an idempotent wrapper that keeps the solver inputs and returned
metadata synchronized.
"""

from __future__ import annotations

# pylint: disable=too-many-arguments,too-many-branches

import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import pyrecest_global_assignment as _global_assignment

SessionEdge = tuple[int, int]
_PATCH_ATTR = "_bayescatrack_global_assignment_input_validation_patch"


def install_global_assignment_input_validation() -> None:
    """Install an idempotent validator around the global-assignment helper."""

    original = _global_assignment.solve_global_assignment_from_pairwise_costs
    if getattr(original, _PATCH_ATTR, False):
        return

    def solve_global_assignment_from_pairwise_costs(
        pairwise_costs: Mapping[SessionEdge, np.ndarray],
        *,
        session_sizes: Sequence[int],
        session_edges: Sequence[SessionEdge] | None = None,
        start_cost: float = 5.0,
        end_cost: float = 5.0,
        gap_penalty: float = 1.0,
        cost_threshold: float | None = 6.0,
    ) -> _global_assignment.GlobalAssignmentRun:
        sizes = _normalize_session_sizes(session_sizes)
        normalized_costs = _normalize_pairwise_costs(
            pairwise_costs,
            session_sizes=sizes,
        )
        normalized_edges = _normalize_session_edges(
            normalized_costs,
            session_edges=session_edges,
            session_count=len(sizes),
        )
        return original(
            normalized_costs,
            session_sizes=sizes,
            session_edges=normalized_edges,
            start_cost=start_cost,
            end_cost=end_cost,
            gap_penalty=gap_penalty,
            cost_threshold=cost_threshold,
        )

    setattr(solve_global_assignment_from_pairwise_costs, _PATCH_ATTR, True)
    setattr(
        solve_global_assignment_from_pairwise_costs,
        "_bayescatrack_original",
        original,
    )
    _global_assignment.solve_global_assignment_from_pairwise_costs = (  # type: ignore[assignment]
        solve_global_assignment_from_pairwise_costs
    )


def _normalize_pairwise_costs(
    pairwise_costs: Mapping[Any, Any],
    *,
    session_sizes: tuple[int, ...],
) -> dict[SessionEdge, np.ndarray]:
    normalized: dict[SessionEdge, np.ndarray] = {}
    for raw_edge, raw_cost_matrix in pairwise_costs.items():
        edge = _normalize_session_edge(
            raw_edge,
            session_count=len(session_sizes),
            context="pairwise_costs",
        )
        if edge in normalized:
            raise ValueError(
                f"pairwise_costs contains duplicate session edge {edge!r} after normalization"
            )

        cost_matrix = np.asarray(raw_cost_matrix, dtype=float)
        if cost_matrix.ndim != 2:
            raise ValueError(
                f"pairwise cost matrix for edge {edge!r} must be two-dimensional"
            )
        expected_shape = (session_sizes[edge[0]], session_sizes[edge[1]])
        if cost_matrix.shape != expected_shape:
            raise ValueError(
                f"pairwise cost matrix for edge {edge!r} must have shape "
                f"{expected_shape}, got {cost_matrix.shape}"
            )
        normalized[edge] = cost_matrix
    return normalized


def _normalize_session_edges(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    session_edges: Sequence[Any] | None,
    session_count: int,
) -> tuple[SessionEdge, ...]:
    if session_edges is None:
        return tuple(sorted(pairwise_costs))

    normalized = tuple(
        _normalize_session_edge(edge, session_count=session_count, context="session_edges")
        for edge in session_edges
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("session_edges contains duplicate session edges")

    configured_edges = set(normalized)
    actual_edges = set(pairwise_costs)
    if configured_edges != actual_edges:
        missing_costs = tuple(edge for edge in normalized if edge not in actual_edges)
        unlisted_costs = tuple(
            edge for edge in sorted(actual_edges) if edge not in configured_edges
        )
        details: list[str] = []
        if missing_costs:
            details.append(f"missing pairwise costs for {missing_costs!r}")
        if unlisted_costs:
            details.append(f"unlisted pairwise costs for {unlisted_costs!r}")
        detail_text = "; ".join(details)
        raise ValueError(
            "session_edges and pairwise_costs must describe the same edges"
            f": {detail_text}"
        )
    return normalized


def _normalize_session_sizes(session_sizes: Sequence[Any]) -> tuple[int, ...]:
    return tuple(
        _coerce_integer_like(value, context="session_sizes", allow_zero=True)
        for value in session_sizes
    )


def _normalize_session_edge(
    edge: Any,
    *,
    session_count: int,
    context: str,
) -> SessionEdge:
    try:
        source_raw, target_raw = edge
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} entries must be (source, target) pairs") from exc

    source = _coerce_integer_like(
        source_raw,
        context=f"{context} source session index",
        allow_zero=True,
    )
    target = _coerce_integer_like(
        target_raw,
        context=f"{context} target session index",
        allow_zero=True,
    )
    if source >= target:
        raise ValueError(f"{context} session edges must point forward in time")
    for endpoint_name, endpoint in {"source": source, "target": target}.items():
        if endpoint >= session_count:
            raise ValueError(
                f"{context} {endpoint_name} session index {endpoint} out of bounds "
                f"for {session_count} sessions"
            )
    return int(source), int(target)


def _coerce_integer_like(value: Any, *, context: str, allow_zero: bool) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be an integer")
    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{context} must be an integer")
        integer_value = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{context} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{context} must be an integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{context} must be an integer")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{context} must be an integer") from exc

    minimum = 0 if allow_zero else 1
    if integer_value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return int(integer_value)


__all__ = ["install_global_assignment_input_validation"]
