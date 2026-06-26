"""Consensus edge priors for ensemble-style multi-session assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._numeric_validation import (
    finite_nonnegative_float,
    finite_positive_float,
    positive_integer,
)

SessionEdge = tuple[int, int]
TrackEdge = tuple[int, int, int, int]


@dataclass(frozen=True)
class ConsensusPriorConfig:
    """Configuration for turning agreement across variants into edge relief."""

    variant_costs: tuple[str, ...] = (
        "registered-iou",
        "registered-shifted-iou",
        "roi-aware-shifted",
    )
    min_votes: int = 2
    relief: float = 0.25
    max_relief: float = 1.0
    ignore_variant_failures: bool = True
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant_costs",
            _normalize_variant_costs(self.variant_costs),
        )
        object.__setattr__(
            self,
            "min_votes",
            positive_integer(self.min_votes, name="min_votes"),
        )
        object.__setattr__(
            self,
            "relief",
            finite_nonnegative_float(self.relief, name="relief"),
        )
        object.__setattr__(
            self,
            "max_relief",
            finite_nonnegative_float(self.max_relief, name="max_relief"),
        )
        object.__setattr__(
            self,
            "ignore_variant_failures",
            _strict_bool(self.ignore_variant_failures, name="ignore_variant_failures"),
        )
        object.__setattr__(
            self,
            "large_cost",
            finite_positive_float(self.large_cost, name="large_cost"),
        )


def consensus_prior_config_from_mapping(
    value: ConsensusPriorConfig | Mapping[str, Any] | None,
) -> ConsensusPriorConfig | None:
    """Normalize optional consensus-prior config values."""

    if value is None:
        return None
    if isinstance(value, ConsensusPriorConfig):
        return value
    return ConsensusPriorConfig(**dict(value))


def edge_votes_from_tracks(
    track_sets: Sequence[Sequence[Mapping[int, int]]],
    *,
    session_edges: Sequence[SessionEdge],
) -> dict[TrackEdge, int]:
    """Count assignment edges that recur across multiple solved variants."""

    edges = tuple((int(source), int(target)) for source, target in session_edges)
    votes: dict[TrackEdge, int] = {}
    for tracks in track_sets:
        seen_this_variant: set[TrackEdge] = set()
        for track in tracks:
            normalized: dict[int, int] = {}
            for session, roi in track.items():
                session_index = int(session)
                roi_index = int(roi)
                if roi_index < 0:
                    # Dense track matrices conventionally use -1 for missing detections.
                    # Missing detections must not vote for consensus edges.
                    continue
                normalized[session_index] = roi_index
            for source, target in edges:
                if source not in normalized or target not in normalized:
                    continue
                seen_this_variant.add(
                    (source, target, normalized[source], normalized[target])
                )
        for edge in seen_this_variant:
            votes[edge] = votes.get(edge, 0) + 1
    return votes


def apply_consensus_edge_priors(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    votes: Mapping[TrackEdge, int],
    *,
    config: ConsensusPriorConfig | Mapping[str, Any] | None = None,
) -> dict[SessionEdge, np.ndarray]:
    """Subtract bounded cost relief from edges supported by enough variants."""

    cfg = consensus_prior_config_from_mapping(config) or ConsensusPriorConfig()
    adjusted: dict[SessionEdge, np.ndarray] = {}
    for edge, matrix in pairwise_costs.items():
        session_edge = (int(edge[0]), int(edge[1]))
        adjusted[session_edge] = _normalize_pairwise_cost_matrix(
            matrix,
            session_edge,
        )
    if cfg.relief <= 0.0 or not votes:
        return adjusted

    for (source, target, source_roi, target_roi), vote_count in votes.items():
        if int(vote_count) < cfg.min_votes:
            continue
        matrix = adjusted.get((int(source), int(target)))
        if matrix is None:
            continue
        source_roi = int(source_roi)
        target_roi = int(target_roi)
        if not (
            0 <= source_roi < matrix.shape[0] and 0 <= target_roi < matrix.shape[1]
        ):
            continue
        old_value = float(matrix[source_roi, target_roi])
        if not np.isfinite(old_value) or old_value >= cfg.large_cost:
            continue
        relief = min(cfg.max_relief, cfg.relief * int(vote_count))
        matrix[source_roi, target_roi] = max(0.0, old_value - relief)
    return adjusted


def _normalize_pairwise_cost_matrix(matrix: Any, edge: SessionEdge) -> np.ndarray:
    try:
        costs = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"pairwise cost matrix for session edge {edge!r} must be numeric"
        ) from exc
    if costs.ndim != 2:
        raise ValueError(
            f"pairwise cost matrix for session edge {edge!r} must be two-dimensional"
        )
    return costs.copy()


def _normalize_variant_costs(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw_values = tuple(
            token.strip() for token in values.split(",") if token.strip()
        )
    else:
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise ValueError("variant_costs must be a sequence of names") from exc
    variant_costs = tuple(str(value).strip() for value in raw_values)
    if not variant_costs or any(not value for value in variant_costs):
        raise ValueError("variant_costs must contain non-empty names")
    return variant_costs


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


__all__ = (
    "ConsensusPriorConfig",
    "TrackEdge",
    "consensus_prior_config_from_mapping",
    "edge_votes_from_tracks",
    "apply_consensus_edge_priors",
)
