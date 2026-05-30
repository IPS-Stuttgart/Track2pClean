from __future__ import annotations

from bayescatrack.experiments.track2p_policy_component_audit import ComponentCleanupConfig
from bayescatrack.experiments.track2p_policy_conflict_component_cleanup import (
    ConflictAugmentedCleanupConfig,
    mark_conflict_augmented_splits,
)


def test_conflict_augmented_split_respects_relaxed_complete_track_guard() -> None:
    row = {
        "component_score": 1.10,
        "would_split_at_weakest_edge": 0,
        "weakest_bridge_session_a": 1,
        "total_sessions": 4,
        "n_sessions": 3,
        "is_complete_track": 0,
        "n_conflicting_edges": 1,
        "n_same_predicted_edges": 0,
        "applied_split": 0,
    }

    guarded = mark_conflict_augmented_splits(
        [row],
        config=ConflictAugmentedCleanupConfig(
            component_config=ComponentCleanupConfig(split_risk_threshold=1.50),
            conflicting_observation_bonus=0.50,
            duplicate_edge_bonus=0.0,
            min_base_risk=0.25,
        ),
    )
    relaxed = mark_conflict_augmented_splits(
        [row],
        config=ConflictAugmentedCleanupConfig(
            component_config=ComponentCleanupConfig(
                split_risk_threshold=1.50,
                require_complete_track=False,
            ),
            conflicting_observation_bonus=0.50,
            duplicate_edge_bonus=0.0,
            min_base_risk=0.25,
        ),
    )

    assert guarded[0]["conflict_augmented_extra_split"] == 0
    assert guarded[0]["applied_split"] == 0
    assert relaxed[0]["conflict_augmented_extra_split"] == 1
    assert relaxed[0]["applied_split"] == 1
