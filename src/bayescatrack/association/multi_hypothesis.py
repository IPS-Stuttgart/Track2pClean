"""Multi-hypothesis edge and track utilities for ambiguous ROI tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ._numeric_validation import finite_nonnegative_float as _finite_nonnegative_float
from ._numeric_validation import integer as _integer
from ._numeric_validation import positive_integer as _positive_integer
from ._numeric_validation import probability as _probability

Edge = tuple[int, int, int, int]


@dataclass(frozen=True)
class HypothesisConfig:
    """Beam-search and consensus options."""

    edge_top_k: int = 3
    beam_width: int = 128
    max_edge_cost: float | None = 6.0
    min_consensus_votes: int = 2
    fill_value: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "edge_top_k", _positive_integer(self.edge_top_k, name="edge_top_k")
        )
        object.__setattr__(
            self, "beam_width", _positive_integer(self.beam_width, name="beam_width")
        )
        object.__setattr__(
            self,
            "min_consensus_votes",
            _positive_integer(
                self.min_consensus_votes,
                name="min_consensus_votes",
            ),
        )
        if self.max_edge_cost is not None:
            object.__setattr__(
                self,
                "max_edge_cost",
                _finite_nonnegative_float(self.max_edge_cost, name="max_edge_cost"),
            )
        object.__setattr__(
            self, "fill_value", _integer(self.fill_value, name="fill_value")
        )


@dataclass(frozen=True)
class TrackHypothesis:
    """One candidate track and its accumulated cost."""

    row: tuple[int, ...]
    cost: float


def top_k_edge_candidates(
    cost_matrix: Any,
    *,
    edge: tuple[int, int],
    row_top_k: int = 3,
    max_cost: float | None = None,
) -> tuple[Edge, ...]:
    """Return top-k target candidates for each source row on one session edge."""

    row_top_k = _positive_integer(row_top_k, name="row_top_k")
    if max_cost is not None:
        max_cost = _finite_nonnegative_float(max_cost, name="max_cost")
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    source_session, target_session = _validated_session_edge(edge, name="edge")
    candidates: list[Edge] = []
    for row_index, row in enumerate(costs):
        finite = np.isfinite(row)
        if max_cost is not None:
            finite &= row <= max_cost
        cols = np.flatnonzero(finite)
        if cols.size == 0:
            continue
        ordered = cols[np.argsort(row[cols])[:row_top_k]]
        candidates.extend(
            (source_session, target_session, int(row_index), int(col_index))
            for col_index in ordered
        )
    return tuple(candidates)


def candidate_edge_map(
    pairwise_costs: Mapping[tuple[int, int], Any],
    roi_indices_by_session: Sequence[Sequence[int]],
    *,
    config: HypothesisConfig | None = None,
) -> dict[tuple[int, int], list[tuple[int, int, float]]]:
    """Return top-k candidate edges per source ROI for each session edge."""

    cfg = config or HypothesisConfig()
    output: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for edge, matrix_values in pairwise_costs.items():
        source_session, target_session = _validated_session_edge(edge, name="edge")
        matrix = np.asarray(matrix_values, dtype=float)
        if source_session >= len(roi_indices_by_session) or target_session >= len(
            roi_indices_by_session
        ):
            raise ValueError("edge references a missing ROI-index session")
        source_indices = _validated_index_vector(
            roi_indices_by_session[source_session], name="source ROI indices"
        )
        target_indices = _validated_index_vector(
            roi_indices_by_session[target_session], name="target ROI indices"
        )
        if matrix.shape != (source_indices.size, target_indices.size):
            raise ValueError(
                f"Cost matrix shape for edge {edge} does not match ROI indices"
            )
        rows: list[tuple[int, int, float]] = []
        for row_index, source_roi in enumerate(source_indices):
            costs = matrix[row_index]
            finite = np.isfinite(costs)
            if cfg.max_edge_cost is not None:
                finite &= costs <= float(cfg.max_edge_cost)
            candidates = np.flatnonzero(finite)
            if candidates.size == 0:
                continue
            ordered = candidates[np.argsort(costs[candidates])[: cfg.edge_top_k]]
            for col_index in ordered:
                rows.append(
                    (
                        int(source_roi),
                        int(target_indices[col_index]),
                        float(costs[col_index]),
                    )
                )
        output[(source_session, target_session)] = rows
    return output


def enumerate_track_hypotheses(
    session_names: Sequence[str],
    edge_candidates: Mapping[tuple[int, int], Sequence[tuple[int, int, float]]],
    *,
    start_roi_indices: Sequence[int],
    config: HypothesisConfig | None = None,
) -> list[TrackHypothesis]:
    """Beam-search track hypotheses over consecutive session edges."""

    cfg = config or HypothesisConfig()
    n_sessions = len(tuple(session_names))
    by_edge_source: dict[tuple[int, int], dict[int, list[tuple[int, float]]]] = {}
    for edge, candidates in edge_candidates.items():
        validated_edge = _validated_session_edge(edge, name="edge")
        source_lookup: dict[int, list[tuple[int, float]]] = {}
        for source_roi, target_roi, cost in candidates:
            source_lookup.setdefault(
                _validated_index(source_roi, name="source ROI index"), []
            ).append(
                (
                    _validated_index(target_roi, name="target ROI index"),
                    float(cost),
                )
            )
        by_edge_source[validated_edge] = source_lookup

    hypotheses = [
        TrackHypothesis(row=(_validated_index(roi, name="start ROI index"),), cost=0.0)
        for roi in start_roi_indices
    ]
    for session_index in range(n_sessions - 1):
        lookup = by_edge_source.get((session_index, session_index + 1), {})
        expanded: list[TrackHypothesis] = []
        for hypothesis in hypotheses:
            source_roi = hypothesis.row[-1]
            if source_roi == cfg.fill_value:
                expanded.append(
                    TrackHypothesis(
                        row=(*hypothesis.row, cfg.fill_value),
                        cost=hypothesis.cost,
                    )
                )
                continue
            candidates = lookup.get(int(source_roi), ())
            if not candidates:
                expanded.append(
                    TrackHypothesis(
                        row=(*hypothesis.row, cfg.fill_value),
                        cost=hypothesis.cost,
                    )
                )
                continue
            for target_roi, edge_cost in candidates:
                expanded.append(
                    TrackHypothesis(
                        row=(*hypothesis.row, int(target_roi)),
                        cost=float(hypothesis.cost + edge_cost),
                    )
                )
        expanded.sort(key=lambda item: (item.cost, item.row))
        hypotheses = expanded[: cfg.beam_width]
    return hypotheses


def consensus_edges(
    track_matrices: Sequence[Any],
    *,
    min_votes: int | None = 2,
    min_support_fraction: float | None = None,
    fill_value: int = -1,
) -> dict[Edge, int]:
    """Return edges that appear in enough prediction matrices or edge sets."""

    inputs = tuple(track_matrices)
    fill_value = _integer(fill_value, name="fill_value")
    if min_support_fraction is not None:
        support_fraction = _probability(
            min_support_fraction,
            name="min_support_fraction",
            allow_zero=False,
        )
        threshold = int(np.ceil(support_fraction * len(inputs)))
    else:
        threshold = (
            1 if min_votes is None else _positive_integer(min_votes, name="min_votes")
        )
    counts: dict[Edge, int] = {}
    for matrix_values in inputs:
        raw_matrix = _two_dimensional_object_array(
            matrix_values, name="track matrices or edge sets"
        )
        matrix = _validated_integer_matrix(raw_matrix, name="track matrices or edge sets")
        seen_this_model: set[Edge] = set()
        if _is_edge_set_input(matrix_values, raw_matrix):
            seen_this_model.update(
                _validated_edge(row, name="edge set") for row in matrix
            )
        else:
            for row in matrix:
                for session_index in range(matrix.shape[1] - 1):
                    a = int(row[session_index])
                    b = int(row[session_index + 1])
                    if a == fill_value or b == fill_value:
                        continue
                    if a < 0 or b < 0:
                        raise ValueError(
                            "track matrices or edge sets entries must be non-negative ROI indices or the fill value"
                        )
                    seen_this_model.add((session_index, session_index + 1, a, b))
        for edge in seen_this_model:
            counts[edge] = counts.get(edge, 0) + 1
    return {edge: votes for edge, votes in counts.items() if votes >= threshold}


def _is_edge_set_input(matrix_values: Any, matrix: np.ndarray) -> bool:
    """Return whether a 2D input is an explicit ``(s, t, i, j)`` edge set."""

    if matrix.shape[1] != 4:
        return False
    if isinstance(matrix_values, np.ndarray):
        # A dense ndarray with four columns is ambiguous with a four-session
        # track matrix. Keep ndarray inputs on the established track-matrix path.
        return False
    if not isinstance(matrix_values, tuple):
        # Python lists are the natural representation of track matrices in tests,
        # scripts, and JSON-loaded manifests. Do not reinterpret four-session
        # lists as explicit edge sets.
        return False
    return all(isinstance(row, tuple) and len(row) == 4 for row in matrix_values)


def hypotheses_to_matrix(hypotheses: Sequence[TrackHypothesis]) -> np.ndarray:
    """Convert track hypotheses to a dense integer matrix."""

    if not hypotheses:
        return np.zeros((0, 0), dtype=int)
    width = len(hypotheses[0].row)
    return np.asarray(
        [hyp.row for hyp in hypotheses if len(hyp.row) == width], dtype=int
    )


def edge_union_costs(edge_sets: Sequence[Mapping[Edge, int]]) -> dict[Edge, float]:
    """Return simple consensus costs where more votes means lower cost."""

    votes: dict[Edge, int] = {}
    for edge_set in edge_sets:
        for edge, vote_count in edge_set.items():
            validated_edge = _validated_edge(edge, name="edge")
            votes[validated_edge] = votes.get(validated_edge, 0) + _positive_integer(
                vote_count, name="vote_count"
            )
    return {edge: 1.0 / max(vote_count, 1) for edge, vote_count in votes.items()}


def _validated_session_edge(edge: Any, *, name: str) -> tuple[int, int]:
    try:
        components = tuple(edge)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two entries") from exc
    if len(components) != 2:
        raise ValueError(f"{name} must contain two entries")
    return (
        _validated_index(components[0], name=f"{name} source session"),
        _validated_index(components[1], name=f"{name} target session"),
    )


def _validated_edge(edge: Any, *, name: str) -> Edge:
    try:
        components = tuple(edge)
    except TypeError as exc:
        raise ValueError(f"{name} must contain four entries") from exc
    if len(components) != 4:
        raise ValueError(f"{name} must contain four entries")
    return (
        _validated_index(components[0], name=f"{name} source session"),
        _validated_index(components[1], name=f"{name} target session"),
        _validated_index(components[2], name=f"{name} source ROI"),
        _validated_index(components[3], name=f"{name} target ROI"),
    )


def _validated_index(value: Any, *, name: str) -> int:
    try:
        index = _integer(value, name=name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if index < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return index


def _two_dimensional_object_array(values: Any, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(f"{name} must be two-dimensional") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return matrix


def _validated_integer_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    output = np.empty(matrix.shape, dtype=int)
    for index in np.ndindex(matrix.shape):
        try:
            output[index] = _integer(matrix[index], name=f"{name} entry")
        except ValueError as exc:
            raise ValueError(f"{name} entries must be integers") from exc
    return output


def _validated_index_vector(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(f"{name} must be one-dimensional") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    output = np.empty(array.shape, dtype=int)
    for index, value in enumerate(array):
        output[index] = _validated_index(value, name=f"{name} entry")
    return output
