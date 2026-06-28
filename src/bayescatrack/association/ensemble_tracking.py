"""Consensus utilities for ensembling multiple track matrices."""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
from bayescatrack.evaluation.complete_track_scores import normalize_track_matrix
from bayescatrack.matching import build_track_rows_from_matches

TrackEdge = tuple[int, int, int, int]
_MISSING_OBSERVATION_STRINGS = frozenset({"", "-", "none", "nan", "null"})


def track_matrix_edge_counter(
    track_matrix: Any,
    *,
    session_pairs: Sequence[tuple[int, int]] | None = None,
) -> Counter[TrackEdge]:
    """Return a multiset of ROI identity edges from one track matrix."""

    matrix = _normalize_track_matrix_for_edges(track_matrix)
    pairs = _session_pairs(matrix.shape[1], session_pairs)
    counter: Counter[TrackEdge] = Counter()
    for session_a, session_b in pairs:
        for row in matrix:
            roi_a = _roi_index_or_none(row[session_a])
            roi_b = _roi_index_or_none(row[session_b])
            if roi_a is not None and roi_b is not None:
                counter[(session_a, session_b, roi_a, roi_b)] += 1
    return counter


def consensus_edge_counter(
    track_matrices: Sequence[Any],
    *,
    min_votes: int = 2,
    session_pairs: Sequence[tuple[int, int]] | None = None,
) -> Counter[TrackEdge]:
    """Return edges supported by at least ``min_votes`` model variants."""

    min_votes = _coerce_positive_integer(min_votes, name="min_votes")
    vote_counter: Counter[TrackEdge] = Counter()
    for matrix in track_matrices:
        vote_counter.update(
            track_matrix_edge_counter(matrix, session_pairs=session_pairs).keys()
        )
    return Counter(
        {edge: votes for edge, votes in vote_counter.items() if votes >= min_votes}
    )


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
    start_session_index = _coerce_integer_value(
        start_session_index,
        name="start_session_index",
    )
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


def _normalize_track_matrix_for_edges(track_matrix: Any) -> np.ndarray:
    """Normalize track matrices without silently dropping malformed ROI labels."""

    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must have shape (n_tracks, n_sessions)")
    normalized = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        roi_index = _roi_index_or_none(value)
        normalized[index] = -1 if roi_index is None else roi_index
    return normalize_track_matrix(normalized)


def _roi_index_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed_text = _parse_roi_index_text(value)
        if parsed_text is None:
            return None
        roi_index = parsed_text
    elif isinstance(value, (bool, np.bool_)):
        raise ValueError(f"track matrix contains boolean ROI index: {value!r}")
    elif isinstance(value, (int, np.integer)):
        roi_index = int(value)
    elif isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"track matrix contains non-integer ROI index: {value!r}")
        roi_index = int(value)
    elif isinstance(value, Decimal):
        roi_index = _parse_decimal_roi_index(value)
    elif isinstance(value, Fraction):
        roi_index = _parse_fraction_roi_index(value)
    else:
        try:
            roi_index = operator.index(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                f"track matrix contains non-integer ROI index: {value!r}"
            ) from exc
    if roi_index < 0:
        return None
    return int(roi_index)


def _parse_roi_index_text(value: str) -> int | None:
    stripped = value.strip()
    if stripped.casefold() in _MISSING_OBSERVATION_STRINGS:
        return None
    try:
        return int(stripped, 10)
    except ValueError:
        pass
    try:
        numeric_value = float(stripped)
    except ValueError as exc:
        raise ValueError(
            f"track matrix contains non-integer ROI index: {value!r}"
        ) from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"track matrix contains non-integer ROI index: {value!r}")
    return int(numeric_value)


def _parse_decimal_roi_index(value: Decimal) -> int:
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"track matrix contains non-integer ROI index: {value!r}")
    return int(value)


def _parse_fraction_roi_index(value: Fraction) -> int:
    if value.denominator != 1:
        raise ValueError(f"track matrix contains non-integer ROI index: {value!r}")
    return int(value.numerator)


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
    start_session_index = _coerce_integer_value(
        start_session_index,
        name="start_session_index",
    )
    values: set[int] = set()
    for matrix_like in track_matrices:
        matrix = _normalize_track_matrix_for_edges(matrix_like)
        if start_session_index < 0 or start_session_index >= matrix.shape[1]:
            raise IndexError("start_session_index out of bounds")
        for value in matrix[:, start_session_index]:
            roi_index = _roi_index_or_none(value)
            if roi_index is not None:
                values.add(roi_index)
    return tuple(sorted(values))


def _session_pairs(
    n_sessions: int, pairs: Sequence[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return tuple((i, i + 1) for i in range(max(0, n_sessions - 1)))
    normalized = tuple(
        (
            _coerce_integer_value(a, name="session_pairs"),
            _coerce_integer_value(b, name="session_pairs"),
        )
        for a, b in pairs
    )
    for a, b in normalized:
        if a < 0 or b >= n_sessions or a >= b:
            raise ValueError("session_pairs must be forward pairs within matrix width")
    return normalized


def _coerce_positive_integer(value: object, *, name: str) -> int:
    normalized = _coerce_integer_value(value, name=name)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _coerce_integer_value(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return int(value)
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = (
    "TrackEdge",
    "consensus_edge_counter",
    "consensus_track_rows",
    "track_matrix_edge_counter",
)
