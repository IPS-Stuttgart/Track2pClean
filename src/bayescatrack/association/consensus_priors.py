"""Consensus edge priors for ensemble-style multi-session assignment."""

from __future__ import annotations

import operator
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
        object.__setattr__(
            self,
            "variant_costs",
            _variant_cost_tuple(self.variant_costs, "variant_costs"),
        )
        object.__setattr__(
            self, "min_votes", _positive_int_value(self.min_votes, "min_votes")
        )
        object.__setattr__(
            self,
            "relief",
            _finite_nonnegative_float(self.relief, "relief"),
        )
        object.__setattr__(
            self,
            "max_relief",
            _finite_nonnegative_float(self.max_relief, "max_relief"),
        )
        object.__setattr__(
            self,
            "large_cost",
            _finite_positive_float(self.large_cost, "large_cost"),
        )
        object.__setattr__(
            self,
            "ignore_variant_failures",
            _bool_value(self.ignore_variant_failures, "ignore_variant_failures"),
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
        votes_for_edge = _nonnegative_int_value(vote_count, "vote_count")
        if votes_for_edge < cfg.min_votes:
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
        relief = min(cfg.max_relief, cfg.relief * votes_for_edge)
        matrix[source_roi, target_roi] = max(0.0, old_value - relief)
    return adjusted


def _variant_cost_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        try:
            raw_values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a non-empty sequence") from exc
    output: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            raise ValueError(f"{name} entries must be non-empty strings")
        token = raw.strip()
        if not token:
            raise ValueError(f"{name} entries must be non-empty strings")
        output.append(token)
    if not output:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(output)


def _positive_int_value(value: Any, name: str) -> int:
    integer_value = _integer_value(value, name)
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive")
    return integer_value


def _nonnegative_int_value(value: Any, name: str) -> int:
    integer_value = _integer_value(value, name)
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer_value


def _integer_value(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        integer_value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return int(integer_value)


def _finite_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _finite_positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite value")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return numeric


def _bool_value(value: Any, name: str) -> bool:
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
