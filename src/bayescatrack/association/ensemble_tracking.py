"""Consensus utilities for ensembling multiple track matrices."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from bayescatrack.evaluation.complete_track_scores import normalize_track_matrix
from bayescatrack.matching import build_track_rows_from_matches


TrackEdge = tuple[int, int, int, int]


def track_matrix_edge_counter(
    track_matrix: Any,
    *,
    session_pairs: Sequence[tuple[int, int]] | None = None,
) -> Counter[TrackEdge]:
    """Return a multiset of ROI identity edges from one track matrix."""

    matrix = normalize_track_matrix(track_matrix)
    pairs = _session_pairs(matrix.shape[1], session_pairs)
    counter: Counter[TrackEdge] = Counter()
    for session_a, session_b in pairs:
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is not None and roi_b is not None:
                counter[(session_a, session_b, int(roi_a), int(roi_b))] += 1
    return counter


def consensus_edge_counter(
    track_matrices: Sequence[Any],
    *,
    min_votes: int = 2,
    session_pairs: Sequence[tuple[int, int]] | None = None,
) -> Counter[TrackEdge]:
    """Return edges supported by at least ``min_votes`` model variants."""

    if min_votes <= 0:
        raise ValueError("min_votes must be positive")
    vote_counter: Counter[TrackEdge] = Counter()
    for matrix in track_matrices:
        vote_counter.update(track_matrix_edge_counter(matrix, session_pairs=session_pairs).keys())
    return Counter({edge: votes for edge, votes in vote_counter.items() if votes >= min_votes})


def consensus_track_rows(
    session_names: Sequence[str],
    track_matrices: Sequence[Any],
    *,
    min_votes: int = 2,
    start_roi_indices: Sequence[int] | None = None,
    start_session_index: int = 0,
    fill_value: int = -1,
) -> np.ndarray:
    """Stitch consensus consecutive edges into a track matrix."""

    session_names = tuple(str(name) for name in session_names)
    consecutive_pairs = tuple((i, i + 1) for i in range(max(0, len(session_names) - 1)))
    edges = consensus_edge_counter(
        track_matrices,
        min_votes=min_votes,
        session_pairs=consecutive_pairs,
    )
    matches = []
    for session_a, session_b in consecutive_pairs:
        mapping = _one_to_one_edge_mapping(
            edge for edge in edges if edge[0] == session_a and edge[1] == session_b
        )
        matches.append(mapping)
    if start_roi_indices is None:
        start_roi_indices = _start_indices_from_matrices(
            track_matrices,
            start_session_index=start_session_index,
        )
    return build_track_rows_from_matches(
        session_names,
        matches,
        start_roi_indices=start_roi_indices,
        start_session_index=start_session_index,
        fill_value=fill_value,
    )


def _one_to_one_edge_mapping(edges: Sequence[TrackEdge] | Any) -> dict[int, int]:
    by_source: dict[int, list[int]] = defaultdict(list)
    by_target: dict[int, list[int]] = defaultdict(list)
    for _session_a, _session_b, roi_a, roi_b in edges:
        by_source[int(roi_a)].append(int(roi_b))
        by_target[int(roi_b)].append(int(roi_a))
    mapping: dict[int, int] = {}
    for roi_a, targets in by_source.items():
        if len(targets) != 1:
            continue
        roi_b = targets[0]
        if len(by_target[roi_b]) == 1:
            mapping[roi_a] = roi_b
    return mapping


def _start_indices_from_matrices(
    track_matrices: Sequence[Any], *, start_session_index: int
) -> tuple[int, ...]:
    values: set[int] = set()
    for matrix_like in track_matrices:
        matrix = normalize_track_matrix(matrix_like)
        if start_session_index < 0 or start_session_index >= matrix.shape[1]:
            raise IndexError("start_session_index out of bounds")
        for value in matrix[:, start_session_index]:
            if value is not None:
                values.add(int(value))
    return tuple(sorted(values))


def _session_pairs(
    n_sessions: int, pairs: Sequence[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return tuple((i, i + 1) for i in range(max(0, n_sessions - 1)))
    normalized = tuple((int(a), int(b)) for a, b in pairs)
    for a, b in normalized:
        if a < 0 or b >= n_sessions or a >= b:
            raise ValueError("session_pairs must be forward pairs within matrix width")
    return normalized


__all__ = (
    "TrackEdge",
    "consensus_edge_counter",
    "consensus_track_rows",
    "track_matrix_edge_counter",
)
