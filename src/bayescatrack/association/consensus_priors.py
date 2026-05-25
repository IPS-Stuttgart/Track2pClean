"""Consensus edge priors for ensemble-style multi-session assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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
        if self.min_votes < 1:
            raise ValueError("min_votes must be positive")
        if self.relief < 0.0 or self.max_relief < 0.0:
            raise ValueError("consensus relief values must be non-negative")
        if self.large_cost <= 0.0 or not np.isfinite(self.large_cost):
            raise ValueError("large_cost must be a positive finite value")
        object.__setattr__(
            self, "variant_costs", tuple(str(value) for value in self.variant_costs)
        )


def consensus_prior_config_from_mapping(
    value: ConsensusPriorConfig | Mapping[str, Any] | None,
) -> ConsensusPriorConfig | None:
    """Normalize optional consensus-prior config values."""

    if value is None:
        return None
    if isinstance(value, ConsensusPriorConfig):
        return value
    payload = dict(value)
    if "variant_costs" in payload:
        raw = payload["variant_costs"]
        if isinstance(raw, str):
            payload["variant_costs"] = tuple(
                token.strip() for token in raw.split(",") if token.strip()
            )
        else:
            payload["variant_costs"] = tuple(raw)
    return ConsensusPriorConfig(**payload)


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
    adjusted = {
        (int(edge[0]), int(edge[1])): np.asarray(matrix, dtype=float).copy()
        for edge, matrix in pairwise_costs.items()
    }
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


__all__ = (
    "ConsensusPriorConfig",
    "TrackEdge",
    "consensus_prior_config_from_mapping",
    "edge_votes_from_tracks",
    "apply_consensus_edge_priors",
)
