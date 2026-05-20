"""Heuristic error taxonomy for Track2p-style track matrices."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.advanced_roi_components import pairwise_cost_margin_components
from bayescatrack.evaluation.complete_track_scores import normalize_track_matrix

TrackLink = tuple[int, int, int, int]
SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class LinkErrorRecord:
    """One classified pairwise link error."""

    kind: str
    category: str
    session_a: int
    session_b: int
    roi_a: int
    roi_b: int
    cost: float | None = None
    row_rank_cost: float | None = None
    column_rank_cost: float | None = None

    def to_row(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "kind": self.kind,
            "category": self.category,
            "session_a": int(self.session_a),
            "session_b": int(self.session_b),
            "roi_a": int(self.roi_a),
            "roi_b": int(self.roi_b),
        }
        if self.cost is not None:
            row["cost"] = float(self.cost)
        if self.row_rank_cost is not None:
            row["row_rank_cost"] = float(self.row_rank_cost)
        if self.column_rank_cost is not None:
            row["column_rank_cost"] = float(self.column_rank_cost)
        return row


@dataclass(frozen=True)
class TrackErrorTaxonomyReport:
    """Detailed and aggregate classified track errors."""

    records: tuple[LinkErrorRecord, ...]

    def summary(self) -> dict[str, int]:
        counts = Counter(record.category for record in self.records)
        by_kind = Counter(record.kind for record in self.records)
        return {
            **{f"taxonomy_{key}": int(value) for key, value in sorted(counts.items())},
            **{f"taxonomy_{key}": int(value) for key, value in sorted(by_kind.items())},
            "taxonomy_total_errors": int(len(self.records)),
        }

    def to_rows(self) -> list[dict[str, float | int | str]]:
        return [record.to_row() for record in self.records]


def classify_track_errors(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[SessionEdge] | None = None,
    pairwise_costs: Mapping[SessionEdge, np.ndarray] | None = None,
    cost_threshold: float | None = None,
    ambiguity_rank_threshold: float = 0.15,
    large_cost: float = 1.0e6,
) -> TrackErrorTaxonomyReport:
    """Classify false-positive and false-negative links with heuristic causes.

    The taxonomy is intentionally conservative.  It works without cost matrices
    and becomes more specific when pairwise costs are supplied in the same ROI
    index space as the track matrices.
    """

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")
    pairs = _session_pairs(predicted.shape[1], session_pairs)
    predicted_links = _track_link_counter(predicted, pairs)
    reference_links = _track_link_counter(reference, pairs)
    predicted_observations = _observation_counter(predicted)
    reference_observations = _observation_counter(reference)

    records: list[LinkErrorRecord] = []
    for link, count in (predicted_links - reference_links).items():
        for _ in range(count):
            records.append(
                _classify_false_positive(
                    link,
                    predicted_observations=predicted_observations,
                    reference_links=reference_links,
                    pairwise_costs=pairwise_costs,
                    cost_threshold=cost_threshold,
                    ambiguity_rank_threshold=ambiguity_rank_threshold,
                    large_cost=large_cost,
                )
            )
    for link, count in (reference_links - predicted_links).items():
        for _ in range(count):
            records.append(
                _classify_false_negative(
                    link,
                    reference_observations=reference_observations,
                    predicted_observations=predicted_observations,
                    pairwise_costs=pairwise_costs,
                    cost_threshold=cost_threshold,
                    ambiguity_rank_threshold=ambiguity_rank_threshold,
                    large_cost=large_cost,
                )
            )
    return TrackErrorTaxonomyReport(records=tuple(records))


def summarize_error_taxonomy(*args: Any, **kwargs: Any) -> dict[str, int]:
    """Return aggregate taxonomy counts for ``classify_track_errors``."""

    return classify_track_errors(*args, **kwargs).summary()


def _classify_false_positive(
    link: TrackLink,
    *,
    predicted_observations: Counter[tuple[int, int]],
    reference_links: Counter[TrackLink],
    pairwise_costs: Mapping[SessionEdge, np.ndarray] | None,
    cost_threshold: float | None,
    ambiguity_rank_threshold: float,
    large_cost: float,
) -> LinkErrorRecord:
    session_a, session_b, roi_a, roi_b = link
    cost_info = _cost_info(
        link,
        pairwise_costs=pairwise_costs,
        large_cost=large_cost,
    )
    category = "false_continuation"
    if predicted_observations[(session_a, roi_a)] > 1 or predicted_observations[(session_b, roi_b)] > 1:
        category = "duplicate_observation"
    elif _shares_endpoint_with_reference(link, reference_links):
        category = "split_merge_or_neighbor_swap"
    elif cost_threshold is not None and cost_info["cost"] is not None and cost_info["cost"] > cost_threshold:
        category = "threshold_too_loose"
    elif _is_ambiguous(cost_info, ambiguity_rank_threshold=ambiguity_rank_threshold):
        category = "ambiguous_candidate"
    return LinkErrorRecord(
        kind="false_positive",
        category=category,
        session_a=session_a,
        session_b=session_b,
        roi_a=roi_a,
        roi_b=roi_b,
        cost=cost_info["cost"],
        row_rank_cost=cost_info["row_rank_cost"],
        column_rank_cost=cost_info["column_rank_cost"],
    )


def _classify_false_negative(
    link: TrackLink,
    *,
    reference_observations: Counter[tuple[int, int]],
    predicted_observations: Counter[tuple[int, int]],
    pairwise_costs: Mapping[SessionEdge, np.ndarray] | None,
    cost_threshold: float | None,
    ambiguity_rank_threshold: float,
    large_cost: float,
) -> LinkErrorRecord:
    session_a, session_b, roi_a, roi_b = link
    cost_info = _cost_info(
        link,
        pairwise_costs=pairwise_costs,
        large_cost=large_cost,
    )
    category = "missed_link"
    if predicted_observations[(session_a, roi_a)] == 0 or predicted_observations[(session_b, roi_b)] == 0:
        category = "roi_missing_from_prediction"
    elif reference_observations[(session_a, roi_a)] > 1 or reference_observations[(session_b, roi_b)] > 1:
        category = "reference_split_merge"
    elif cost_threshold is not None and cost_info["cost"] is not None and cost_info["cost"] > cost_threshold:
        category = "threshold_too_strict"
    elif _is_ambiguous(cost_info, ambiguity_rank_threshold=ambiguity_rank_threshold):
        category = "ambiguous_candidate"
    return LinkErrorRecord(
        kind="false_negative",
        category=category,
        session_a=session_a,
        session_b=session_b,
        roi_a=roi_a,
        roi_b=roi_b,
        cost=cost_info["cost"],
        row_rank_cost=cost_info["row_rank_cost"],
        column_rank_cost=cost_info["column_rank_cost"],
    )


def _cost_info(
    link: TrackLink,
    *,
    pairwise_costs: Mapping[SessionEdge, np.ndarray] | None,
    large_cost: float,
) -> dict[str, float | None]:
    session_a, session_b, roi_a, roi_b = link
    if pairwise_costs is None or (session_a, session_b) not in pairwise_costs:
        return {"cost": None, "row_rank_cost": None, "column_rank_cost": None}
    matrix = np.asarray(pairwise_costs[(session_a, session_b)], dtype=float)
    if roi_a < 0 or roi_b < 0 or roi_a >= matrix.shape[0] or roi_b >= matrix.shape[1]:
        return {"cost": None, "row_rank_cost": None, "column_rank_cost": None}
    margins = pairwise_cost_margin_components(matrix, large_cost=large_cost)
    return {
        "cost": float(matrix[roi_a, roi_b]),
        "row_rank_cost": float(margins["row_rank_cost"][roi_a, roi_b]),
        "column_rank_cost": float(margins["column_rank_cost"][roi_a, roi_b]),
    }


def _is_ambiguous(
    cost_info: Mapping[str, float | None],
    *,
    ambiguity_rank_threshold: float,
) -> bool:
    row_rank = cost_info.get("row_rank_cost")
    column_rank = cost_info.get("column_rank_cost")
    if row_rank is None or column_rank is None:
        return False
    return min(float(row_rank), float(column_rank)) <= float(ambiguity_rank_threshold)


def _shares_endpoint_with_reference(link: TrackLink, reference_links: Counter[TrackLink]) -> bool:
    session_a, session_b, roi_a, roi_b = link
    for ref_session_a, ref_session_b, ref_roi_a, ref_roi_b in reference_links:
        if ref_session_a != session_a or ref_session_b != session_b:
            continue
        if ref_roi_a == roi_a or ref_roi_b == roi_b:
            return True
    return False


def _track_link_counter(matrix: np.ndarray, session_pairs: Iterable[SessionEdge]) -> Counter[TrackLink]:
    counter: Counter[TrackLink] = Counter()
    for session_a, session_b in session_pairs:
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is not None and roi_b is not None:
                counter[(int(session_a), int(session_b), int(roi_a), int(roi_b))] += 1
    return counter


def _observation_counter(matrix: np.ndarray) -> Counter[tuple[int, int]]:
    counter: Counter[tuple[int, int]] = Counter()
    for row in matrix:
        for session_index, roi_index in enumerate(row):
            if roi_index is not None:
                counter[(int(session_index), int(roi_index))] += 1
    return counter


def _session_pairs(n_sessions: int, session_pairs: Iterable[SessionEdge] | None) -> tuple[SessionEdge, ...]:
    if session_pairs is None:
        return tuple((index, index + 1) for index in range(max(n_sessions - 1, 0)))
    pairs = tuple((int(source), int(target)) for source, target in session_pairs)
    for source, target in pairs:
        if source < 0 or target <= source or target >= n_sessions:
            raise ValueError(f"Invalid session pair {(source, target)!r}")
    return pairs


__all__ = [
    "LinkErrorRecord",
    "TrackErrorTaxonomyReport",
    "classify_track_errors",
    "summarize_error_taxonomy",
]
