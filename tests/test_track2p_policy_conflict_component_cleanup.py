from __future__ import annotations

from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_conflict_component_cleanup import (
    ConflictAugmentedCleanupConfig,
    mark_conflict_augmented_splits,
)


def test_conflict_augmented_cleanup_marks_structural_conflict_split() -> None:
    rows = mark_conflict_augmented_splits(
        [
            _component_row(
                component_score=1.10,
                n_conflicting_edges=1,
            )
        ],
        config=ConflictAugmentedCleanupConfig(
            component_config=ComponentCleanupConfig(split_risk_threshold=1.50),
            conflicting_observation_bonus=0.50,
            duplicate_edge_bonus=0.0,
            min_base_risk=0.25,
        ),
    )

    assert rows[0]["would_split_at_weakest_edge"] == 0
    assert rows[0]["would_split_with_conflict_augmentation"] == 1
    assert rows[0]["conflict_augmented_extra_split"] == 1
    assert rows[0]["applied_split"] == 1
    assert abs(float(rows[0]["conflict_augmented_component_score"]) - 1.60) < 1e-12


def test_conflict_augmented_cleanup_requires_structural_evidence() -> None:
    rows = mark_conflict_augmented_splits(
        [_component_row(component_score=1.10)],
        config=ConflictAugmentedCleanupConfig(
            component_config=ComponentCleanupConfig(split_risk_threshold=1.50),
            conflicting_observation_bonus=0.50,
            duplicate_edge_bonus=0.50,
            min_base_risk=0.25,
        ),
    )

    assert rows[0]["would_split_with_conflict_augmentation"] == 0
    assert rows[0]["conflict_augmented_extra_split"] == 0
    assert rows[0]["applied_split"] == 0


def test_conflict_augmented_cleanup_requires_some_base_risk() -> None:
    rows = mark_conflict_augmented_splits(
        [
            _component_row(
                component_score=0.10,
                n_conflicting_edges=1,
                n_same_predicted_edges=1,
            )
        ],
        config=ConflictAugmentedCleanupConfig(
            component_config=ComponentCleanupConfig(split_risk_threshold=1.50),
            conflicting_observation_bonus=1.00,
            duplicate_edge_bonus=1.00,
            min_base_risk=0.25,
        ),
    )

    assert rows[0]["conflict_augmented_component_score"] == 2.10
    assert rows[0]["conflict_augmented_extra_split"] == 0
    assert rows[0]["applied_split"] == 0


def test_conflict_augmented_cleanup_preserves_existing_component_split() -> None:
    rows = mark_conflict_augmented_splits(
        [_component_row(would_split_at_weakest_edge=1)],
        apply_splits=True,
    )

    assert rows[0]["conflict_augmented_extra_split"] == 0
    assert rows[0]["would_split_with_conflict_augmentation"] == 1
    assert rows[0]["applied_split"] == 1


def test_conflict_augmented_cleanup_audit_mode_marks_but_does_not_apply() -> None:
    rows = mark_conflict_augmented_splits(
        [_component_row(component_score=1.10, n_conflicting_edges=1)],
        config=ConflictAugmentedCleanupConfig(
            component_config=ComponentCleanupConfig(split_risk_threshold=1.50),
            conflicting_observation_bonus=0.50,
            duplicate_edge_bonus=0.0,
            min_base_risk=0.25,
        ),
        apply_splits=False,
    )

    assert rows[0]["would_split_with_conflict_augmentation"] == 1
    assert rows[0]["conflict_augmented_extra_split"] == 1
    assert rows[0]["applied_split"] == 0


def _component_row(**overrides: float | int | str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "component_score": 0.0,
        "would_split_at_weakest_edge": 0,
        "weakest_bridge_session_a": 1,
        "total_sessions": 4,
        "n_sessions": 4,
        "is_complete_track": 1,
        "n_conflicting_edges": 0,
        "n_same_predicted_edges": 0,
        "applied_split": 0,
    }
    row.update(overrides)
    return row
