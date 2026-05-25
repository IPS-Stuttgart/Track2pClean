"""Consensus bridge-split helper for Track2p-policy cleanup experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    edge_risk_score,
)
from bayescatrack.experiments.track2p_policy_multisplit_cleanup import (
    apply_ranked_bridge_splits,
    split_track_at_bridges,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
)
from bayescatrack.experiments.track2p_policy_stability_cleanup import Edge

ConsensusMode = Literal[
    "risk-and-stability",
    "risk-or-stability",
    "risk-only",
    "stability-only",
]


@dataclass(frozen=True)
class ConsensusSplitConfig:
    """Controls how edge-risk and threshold-stability evidence are combined."""

    component: ComponentCleanupConfig = ComponentCleanupConfig()
    required_support_votes: int = 2
    max_splits_per_component: int = 2
    mode: ConsensusMode = "risk-and-stability"

    def __post_init__(self) -> None:
        if int(self.required_support_votes) < 1:
            raise ValueError("required_support_votes must be at least 1")
        if int(self.max_splits_per_component) < 1:
            raise ValueError("max_splits_per_component must be at least 1")
        if self.mode not in {
            "risk-and-stability",
            "risk-or-stability",
            "risk-only",
            "stability-only",
        }:
            raise ValueError("unsupported consensus mode")


def plan_consensus_bridge_splits(
    predicted_track_matrix: np.ndarray,
    *,
    diagnostics_by_edge: Mapping[tuple[int, int, int], Track2pPolicyLinkDiagnostic],
    support_counts: Mapping[Edge, int],
    config: ConsensusSplitConfig | None = None,
) -> dict[int, tuple[int, ...]]:
    """Plan conservative splits for risky, unstable bridges.

    ``diagnostics_by_edge`` uses ``(session, source_roi, target_roi)`` keys in
    the same ROI-index space as ``predicted_track_matrix``. ``support_counts``
    uses ``(session, next_session, source_roi, target_roi)`` keys as returned by
    the stability-cleanup helper.
    """

    cfg = config or ConsensusSplitConfig()
    predicted = np.asarray(predicted_track_matrix, dtype=int)
    if predicted.ndim != 2:
        raise ValueError("predicted_track_matrix must be two-dimensional")
    split_plan: dict[int, tuple[int, ...]] = {}
    for track_id, row in enumerate(predicted):
        splits = _track_split_indices(row, diagnostics_by_edge, support_counts, cfg)
        if splits:
            split_plan[int(track_id)] = splits
    return split_plan


def apply_consensus_bridge_splits(
    predicted_track_matrix: np.ndarray,
    split_plan: Mapping[int, Sequence[int]],
) -> np.ndarray:
    """Apply a consensus split plan to a prediction matrix."""

    return apply_ranked_bridge_splits(predicted_track_matrix, split_plan)


def _track_split_indices(
    row: np.ndarray,
    diagnostics_by_edge: Mapping[tuple[int, int, int], Track2pPolicyLinkDiagnostic],
    support_counts: Mapping[Edge, int],
    cfg: ConsensusSplitConfig,
) -> tuple[int, ...]:
    if cfg.component.require_complete_track and int(np.sum(row >= 0)) != int(row.size):
        return ()
    candidates: list[tuple[int, float, int]] = []
    for session_index in range(max(0, row.size - 1)):
        source = int(row[session_index])
        target = int(row[session_index + 1])
        if source < 0 or target < 0:
            continue
        diagnostic = diagnostics_by_edge.get((session_index, source, target))
        risk = edge_risk_score(diagnostic, config=cfg.component)
        support = int(support_counts.get((session_index, session_index + 1, source, target), 0))
        risky = risk >= cfg.component.split_risk_threshold and risk > cfg.component.split_penalty
        unstable = support < int(cfg.required_support_votes)
        if _passes_mode(risky, unstable, cfg.mode):
            candidates.append((support, risk, session_index))

    selected: list[int] = []
    for _, _, split_index in sorted(candidates, key=lambda item: (item[0], -item[1], item[2])):
        if len(selected) >= int(cfg.max_splits_per_component):
            break
        proposed = sorted((*selected, split_index))
        if _fragments_have_min_observations(row, proposed, cfg.component.min_side_observations):
            selected.append(split_index)
    return tuple(sorted(selected))


def _passes_mode(risky: bool, unstable: bool, mode: ConsensusMode) -> bool:
    if mode == "risk-and-stability":
        return bool(risky and unstable)
    if mode == "risk-or-stability":
        return bool(risky or unstable)
    if mode == "risk-only":
        return bool(risky)
    if mode == "stability-only":
        return bool(unstable)
    raise ValueError(f"unsupported consensus mode: {mode}")


def _fragments_have_min_observations(
    row: np.ndarray, split_indices: Sequence[int], min_observations: int
) -> bool:
    return all(
        int(np.sum(fragment >= 0)) >= int(min_observations)
        for fragment in split_track_at_bridges(row, split_indices)
    )


__all__ = (
    "ConsensusSplitConfig",
    "apply_consensus_bridge_splits",
    "plan_consensus_bridge_splits",
)
