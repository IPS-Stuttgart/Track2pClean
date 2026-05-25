from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_consensus_cleanup import (
    ConsensusSplitConfig,
    apply_consensus_bridge_splits,
    plan_consensus_bridge_splits,
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
        diagnostics_by_edge=diagnostics,  # type: ignore[arg-type]
        support_counts=support,
        config=ConsensusSplitConfig(
            component=ComponentCleanupConfig(split_risk_threshold=1.0),
            required_support_votes=2,
        ),
    )

    assert plan == {0: (1,)}


def test_consensus_risk_only_ablation_and_apply() -> None:
    predicted = np.asarray([[10, 20, 30, 40, 50, 60]], dtype=int)
    diagnostics = {(1, 20, 30): _Diagnostic(), (3, 40, 50): _Diagnostic()}
    support = {(1, 2, 20, 30): 3, (3, 4, 40, 50): 3}

    plan = plan_consensus_bridge_splits(
        predicted,
        diagnostics_by_edge=diagnostics,  # type: ignore[arg-type]
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
