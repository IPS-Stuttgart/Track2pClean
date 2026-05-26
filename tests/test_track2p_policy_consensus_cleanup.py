from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_consensus_cleanup import (
    ConsensusCleanupConfig,
    ConsensusSplitConfig,
    apply_consensus_bridge_splits,
    plan_consensus_bridge_splits,
)
from bayescatrack.experiments.track2p_policy_stability_cleanup import (
    StabilityCleanupConfig,
)


@dataclass(frozen=True)
class _Diagnostic:
    threshold_margin: float = 0.01
    row_margin: float = 0.02
    column_margin: float = 0.03
    centroid_distance: float = 8.0
    area_ratio: float = 0.20


def test_consensus_requires_risk_and_instability() -> None:
    predicted = np.asarray([[10, 20, 30, 40, 50, 60]], dtype=int)
    diagnostics = {(1, 20, 30): _Diagnostic(), (3, 40, 50): _Diagnostic()}
    support = {(1, 2, 20, 30): 1, (3, 4, 40, 50): 3}

    plan = plan_consensus_bridge_splits(
        predicted,
        diagnostics_by_edge=diagnostics,
        support_counts=support,
        config=ConsensusSplitConfig(
            component=ComponentCleanupConfig(split_risk_threshold=1.0),
            required_support_votes=2,
        ),
    )

    assert plan == {0: (1,)}


def test_consensus_preserves_original_track_ids_for_filtered_rows() -> None:
    predicted_eval = np.asarray([[10, 20, 30, 40]], dtype=int)
    diagnostics = {(1, 20, 30): _Diagnostic()}
    support = {(1, 2, 20, 30): 0}

    plan = plan_consensus_bridge_splits(
        predicted_eval,
        diagnostics_by_edge=diagnostics,
        support_counts=support,
        config=ConsensusSplitConfig(
            component=ComponentCleanupConfig(split_risk_threshold=1.0),
            required_support_votes=2,
        ),
        track_ids=(7,),
    )

    assert plan == {7: (1,)}


def test_consensus_optimizes_compatible_splits_not_greedy_central_bridge() -> None:
    predicted = np.asarray([[10, 20, 30, 40, 50, 60]], dtype=int)
    diagnostics = {
        (1, 20, 30): _Diagnostic(centroid_distance=3.2),
        (2, 30, 40): _Diagnostic(centroid_distance=4.0),
        (3, 40, 50): _Diagnostic(centroid_distance=3.2),
    }
    support = {
        (1, 2, 20, 30): 0,
        (2, 3, 30, 40): 0,
        (3, 4, 40, 50): 0,
    }
    component_config = ComponentCleanupConfig(
        centroid_distance_scale=1.0,
        split_risk_threshold=0.5,
        split_penalty=0.0,
        min_side_observations=2,
        threshold_margin_weight=0.0,
        row_margin_weight=0.0,
        column_margin_weight=0.0,
        centroid_distance_weight=1.0,
        area_ratio_weight=0.0,
    )

    plan = plan_consensus_bridge_splits(
        predicted,
        diagnostics_by_edge=diagnostics,
        support_counts=support,
        config=ConsensusSplitConfig(
            component=component_config,
            required_support_votes=1,
            max_splits_per_component=2,
        ),
    )

    assert plan == {0: (1, 3)}


def test_consensus_risk_only_ablation_and_apply() -> None:
    predicted = np.asarray([[10, 20, 30, 40, 50, 60]], dtype=int)
    diagnostics = {(1, 20, 30): _Diagnostic(), (3, 40, 50): _Diagnostic()}
    support = {(1, 2, 20, 30): 3, (3, 4, 40, 50): 3}

    plan = plan_consensus_bridge_splits(
        predicted,
        diagnostics_by_edge=diagnostics,
        support_counts=support,
        config=ConsensusSplitConfig(
            component=ComponentCleanupConfig(split_risk_threshold=1.0),
            required_support_votes=2,
            mode="risk-only",
        ),
    )

    assert plan == {0: (1, 3)}
    np.testing.assert_array_equal(
        apply_consensus_bridge_splits(predicted, plan),
        np.asarray(
            [
                [10, 20, -1, -1, -1, -1],
                [-1, -1, 30, 40, -1, -1],
                [-1, -1, -1, -1, 50, 60],
            ],
            dtype=int,
        ),
    )


def test_consensus_cleanup_config_uses_stability_support_votes() -> None:
    config = ConsensusCleanupConfig(
        stability=StabilityCleanupConfig(
            iou_distance_thresholds=(10.0, 12.0, 14.0),
            base_iou_distance_threshold=12.0,
            min_support_fraction=1.0,
        ),
        max_splits_per_component=3,
        mode="risk-or-stability",
    )

    assert config.split_config.required_support_votes == 3
    assert config.split_config.max_splits_per_component == 3
    assert config.split_config.mode == "risk-or-stability"


def test_consensus_split_config_rejects_invalid_votes() -> None:
    with pytest.raises(ValueError, match="required_support_votes"):
        ConsensusSplitConfig(required_support_votes=0)


def test_consensus_split_config_rejects_invalid_max_splits() -> None:
    with pytest.raises(ValueError, match="max_splits_per_component"):
        ConsensusSplitConfig(max_splits_per_component=0)


def test_consensus_split_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unsupported consensus mode"):
        ConsensusSplitConfig(mode="bad-mode")
