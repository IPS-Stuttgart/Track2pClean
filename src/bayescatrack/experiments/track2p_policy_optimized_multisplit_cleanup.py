"""Optimized guarded multi-bridge cleanup for Track2p-policy tracks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments import track2p_policy_multisplit_cleanup as _base
from bayescatrack.experiments.track2p_policy_component_audit import (
    _component_edges,
    _diagnostics_by_suite2p_edge,
    _normalize_int_track_matrix,
)
from bayescatrack.experiments.track2p_policy_multisplit_cleanup import (
    MultiSplitCleanupConfig,
    _fragments_satisfy_min_observations,
    _multi_split_gain,
)

TRACK2P_POLICY_MULTISPLIT_CLEANUP_METHOD = (
    _base.TRACK2P_POLICY_MULTISPLIT_CLEANUP_METHOD
)
apply_ranked_bridge_splits = _base.apply_ranked_bridge_splits
build_arg_parser = _base.build_arg_parser
split_track_at_bridges = _base.split_track_at_bridges


def plan_ranked_bridge_splits(
    predicted_track_matrix: Any,
    *,
    sessions: Sequence[Track2pSession],
    diagnostics: Sequence[Any],
    config: MultiSplitCleanupConfig | None = None,
    track_ids: Sequence[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return the feasible split subset with maximal total risk gain."""

    config = config or MultiSplitCleanupConfig()
    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    ids = (
        tuple(range(predicted.shape[0]))
        if track_ids is None
        else tuple(int(track_id) for track_id in track_ids)
    )
    if len(ids) != predicted.shape[0]:
        raise ValueError("track_ids must have one entry per predicted track")
    diagnostic_by_edge = _diagnostics_by_suite2p_edge(sessions, diagnostics)
    split_plan: dict[int, tuple[int, ...]] = {}
    for row_index, track in enumerate(predicted):
        selected = _optimal_split_indices_for_track(
            track,
            diagnostic_by_edge=diagnostic_by_edge,
            config=config,
        )
        if selected:
            split_plan[int(ids[row_index])] = selected
    return split_plan


def run_track2p_policy_multisplit_cleanup(*args: Any, **kwargs: Any) -> Any:
    """Run the base multisplit benchmark with this planner."""

    _base.plan_ranked_bridge_splits = plan_ranked_bridge_splits
    return _base.run_track2p_policy_multisplit_cleanup(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Run the base multisplit CLI with this planner."""

    _base.plan_ranked_bridge_splits = plan_ranked_bridge_splits
    return int(_base.main(argv))


def _optimal_split_indices_for_track(
    track: np.ndarray,
    *,
    diagnostic_by_edge: Mapping[tuple[int, int, int], Any],
    config: MultiSplitCleanupConfig,
) -> tuple[int, ...]:
    component_config = config.component
    observed = int(np.sum(track >= 0))
    if component_config.require_complete_track and observed != int(track.size):
        return ()
    candidate_risks: dict[int, float] = {}
    for edge in _component_edges(track, diagnostic_by_edge, config=component_config):
        index = int(edge.session_index)
        risk = float(edge.risk)
        if risk < component_config.split_risk_threshold:
            continue
        if _multi_split_gain(risk, config=component_config) <= 0.0:
            continue
        candidate_risks[index] = max(risk, candidate_risks.get(index, 0.0))
    return _best_feasible_split_subset(track, candidate_risks, config=config)


def _best_feasible_split_subset(
    track: np.ndarray,
    candidate_risks: Mapping[int, float],
    *,
    config: MultiSplitCleanupConfig,
) -> tuple[int, ...]:
    candidate_indices = tuple(sorted(int(index) for index in candidate_risks))
    max_splits = min(int(config.max_splits_per_component), len(candidate_indices))
    best_gain = 0.0
    best_count = 0
    best_selected: tuple[int, ...] = ()
    for count in range(1, max_splits + 1):
        for candidate in combinations(candidate_indices, count):
            if not _fragments_satisfy_min_observations(
                track,
                candidate,
                min_observations=config.component.min_side_observations,
            ):
                continue
            gain = sum(
                _multi_split_gain(
                    float(candidate_risks[index]), config=config.component
                )
                for index in candidate
            )
            if (
                gain > best_gain
                or (gain == best_gain and count > best_count)
                or (
                    gain == best_gain
                    and count == best_count
                    and (not best_selected or candidate < best_selected)
                )
            ):
                best_gain = float(gain)
                best_count = int(count)
                best_selected = tuple(int(index) for index in candidate)
    return best_selected


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
