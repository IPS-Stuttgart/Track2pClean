"""Strict validation for tracking global-link session-edge metadata.

Global tracking stores the solver's session edges in ``global_link_edges`` and
uses the same metadata to build diagnostic link-cost columns.  The implementation
used direct ``int(...)`` coercion, so booleans, fractional floats, and invalid
edge directions could be silently normalized or fail later as unrelated indexing
errors.  This patch validates the metadata before it is stored or used.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_tracking_global_link_edge_validation_patch"


@dataclass(frozen=True)
class _GlobalAssignmentWithValidatedEdges:
    original: Any
    session_edges: tuple[tuple[int, int], ...]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


def install_tracking_global_link_edge_validation() -> None:
    """Install idempotent validation around global tracking edge metadata."""

    from . import tracking as _tracking  # pylint: disable=import-outside-toplevel

    original_post_init = _tracking.SubjectTrackingResult.__post_init__
    original_build_global_links = (
        _tracking._build_global_link_cost_matrices  # pylint: disable=protected-access
    )

    if getattr(original_post_init, _PATCH_MARKER, False) and getattr(
        original_build_global_links,
        _PATCH_MARKER,
        False,
    ):
        return

    @wraps(original_post_init)
    def subject_tracking_result_post_init_with_global_link_edge_validation(
        self: Any,
    ) -> None:
        session_count = _infer_result_session_count(self)
        object.__setattr__(
            self,
            "global_link_edges",
            _normalize_session_edges(
                self.global_link_edges,
                context="global_link_edges",
                session_count=session_count,
            ),
        )
        original_post_init(self)

    @wraps(original_build_global_links)
    def build_global_link_cost_matrices_with_edge_validation(
        global_assignment: Any,
        sessions: Sequence[Any],
        track_rows: Any,
        *,
        fallback_match_results: Sequence[Any],
        fill_value: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...], np.ndarray]:
        sessions = tuple(sessions)
        normalized_edges = _normalize_session_edges(
            global_assignment.session_edges,
            context="global_assignment.session_edges",
            session_count=len(sessions),
            pairwise_costs=getattr(global_assignment, "pairwise_costs", None),
        )
        return original_build_global_links(
            _GlobalAssignmentWithValidatedEdges(global_assignment, normalized_edges),
            sessions,
            track_rows,
            fallback_match_results=fallback_match_results,
            fill_value=fill_value,
        )

    _mark_patch(
        subject_tracking_result_post_init_with_global_link_edge_validation,
        original_post_init,
    )
    _mark_patch(
        build_global_link_cost_matrices_with_edge_validation,
        original_build_global_links,
    )

    _tracking.SubjectTrackingResult.__post_init__ = (  # type: ignore[method-assign]
        subject_tracking_result_post_init_with_global_link_edge_validation
    )
    _tracking._build_global_link_cost_matrices = (  # pylint: disable=protected-access
        build_global_link_cost_matrices_with_edge_validation
    )


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _infer_result_session_count(result: Any) -> int | None:
    try:
        return len(tuple(result.session_names))
    except TypeError:
        return None


def _normalize_session_edges(
    edges: Any,
    *,
    context: str,
    session_count: int | None,
    pairwise_costs: Mapping[tuple[int, int], Any] | None = None,
) -> tuple[tuple[int, int], ...]:
    if isinstance(edges, (str, bytes)):
        raise ValueError(f"{context} must be a sequence of session edge pairs")
    try:
        edge_values = tuple(edges)
    except TypeError as exc:
        raise ValueError(
            f"{context} must be a sequence of session edge pairs"
        ) from exc

    normalized = tuple(
        _normalize_session_edge(
            edge,
            context=f"{context}[{edge_index}]",
            session_count=session_count,
        )
        for edge_index, edge in enumerate(edge_values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context} contains duplicate session edges")

    if pairwise_costs is not None:
        if not isinstance(pairwise_costs, Mapping):
            raise ValueError("global_assignment.pairwise_costs must be a mapping")
        missing_edges = tuple(edge for edge in normalized if edge not in pairwise_costs)
        if missing_edges:
            raise ValueError(
                f"{context} advertises edges missing from pairwise_costs: "
                f"{missing_edges!r}"
            )
    return normalized


def _normalize_session_edge(
    edge: Any,
    *,
    context: str,
    session_count: int | None,
) -> tuple[int, int]:
    if isinstance(edge, (str, bytes)):
        raise ValueError(f"{context} must be a two-item session edge")
    try:
        source_raw, target_raw = edge
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a two-item session edge") from exc

    source = _normalize_session_index(
        source_raw,
        context=f"{context} source session index",
    )
    target = _normalize_session_index(
        target_raw,
        context=f"{context} target session index",
    )
    if source >= target:
        raise ValueError(f"{context} must point forward in time")
    if session_count is not None:
        for endpoint_name, endpoint in {"source": source, "target": target}.items():
            if endpoint >= session_count:
                raise ValueError(
                    f"{context} {endpoint_name} session index {endpoint} out of "
                    f"bounds for {session_count} sessions"
                )
    return source, target


def _normalize_session_index(value: Any, *, context: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{context} must be an integer")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be an integer")
    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{context} must be an integer")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{context} must be an integer") from exc

    integer_value = int(integer_value)
    if integer_value < 0:
        raise ValueError(f"{context} must be non-negative")
    return integer_value


__all__ = ["install_tracking_global_link_edge_validation"]
