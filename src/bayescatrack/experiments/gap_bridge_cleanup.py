"""Utilities for splitting gap-rescued Track2p-policy components.

Gap-rescue propagation can create components with an observed prefix, one or
more missing sessions, and an observed suffix.  Consecutive-edge cleanup does
not see this missing-session bridge, so a false suffix may survive even when the
row is incomplete.  The helpers here promote those explicit temporal gaps to
normal component-cleanup split rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _normalize_int_track_matrix,
)


def promote_gap_bridge_splits(
    predicted_track_matrix: Any,
    component_rows: Sequence[Mapping[str, float | int | str]],
    *,
    config: ComponentCleanupConfig | None = None,
) -> list[dict[str, float | int | str]]:
    """Return component rows with explicit temporal gaps as split candidates.

    The returned rows remain compatible with
    :func:`apply_weakest_bridge_splits`.  Existing higher-risk consecutive
    weakest-bridge decisions are preserved.  Synthetic gap decisions still obey
    the standard ``min_side_observations`` and ``require_complete_track`` guards.
    """

    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    config = config or ComponentCleanupConfig(
        min_side_observations=1,
        require_complete_track=False,
    )
    synthetic_risk = float(config.split_risk_threshold + config.split_penalty + 1.0)
    promoted: list[dict[str, float | int | str]] = []
    for row in component_rows:
        updated = dict(row)
        component_id = int(updated["predicted_track_id"])
        if component_id < 0 or component_id >= predicted.shape[0]:
            promoted.append(updated)
            continue
        split_index = first_observed_gap_bridge(predicted[component_id])
        if split_index < 0:
            promoted.append(updated)
            continue
        left_observations = int(np.sum(predicted[component_id, : split_index + 1] >= 0))
        right_observations = int(np.sum(predicted[component_id, split_index + 1 :] >= 0))
        is_complete_track = bool(updated.get("is_complete_track", 0))
        split_gain = synthetic_risk - float(config.split_penalty)
        would_split = bool(
            left_observations >= config.min_side_observations
            and right_observations >= config.min_side_observations
            and (is_complete_track or not config.require_complete_track)
            and split_gain > 0.0
        )
        existing_split = bool(updated.get("would_split_at_weakest_edge", 0))
        existing_risk = float(updated.get("weakest_bridge_risk", 0.0))
        if existing_split and existing_risk >= synthetic_risk:
            promoted.append(updated)
            continue
        updated.update(
            {
                "component_score": max(
                    float(updated.get("component_score", 0.0)), synthetic_risk
                ),
                "weakest_bridge_session_a": int(split_index),
                "weakest_bridge_session_b": int(split_index + 1),
                "weakest_bridge_source_roi": int(predicted[component_id, split_index]),
                "weakest_bridge_target_roi": -1,
                "weakest_bridge_risk": synthetic_risk,
                "split_gain": split_gain,
                "would_split_at_weakest_edge": int(would_split),
                "gap_bridge_promoted_split": int(would_split),
                "gap_bridge_missing_sessions": int(
                    missing_span_after(predicted[component_id], split_index)
                ),
            }
        )
        promoted.append(updated)
    return promoted


def first_observed_gap_bridge(track: Any) -> int:
    """Return the last observed session before the first observed temporal gap."""

    row = np.asarray(track, dtype=int)
    observed = np.flatnonzero(row >= 0)
    if observed.size < 2:
        return -1
    for left, right in zip(observed[:-1], observed[1:], strict=False):
        if int(right) - int(left) > 1:
            return int(left)
    return -1


def missing_span_after(track: Any, split_index: int) -> int:
    """Return how many consecutive missing sessions follow ``split_index``."""

    row = np.asarray(track, dtype=int)
    span = 0
    for value in row[int(split_index) + 1 :]:
        if int(value) >= 0:
            break
        span += 1
    return span
