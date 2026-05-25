from __future__ import annotations

import numpy as np
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_multisplit_cleanup import (
    MultiSplitCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_optimized_multisplit_cleanup import (
    _best_feasible_split_subset,
)


def test_optimizer_prefers_best_feasible_subset() -> None:
    track = np.asarray([1, 2, 3, 4, 5, 6], dtype=int)
    config = MultiSplitCleanupConfig(
        component=ComponentCleanupConfig(
            split_risk_threshold=1.0,
            split_penalty=0.25,
            min_side_observations=2,
        ),
        max_splits_per_component=2,
    )

    selected = _best_feasible_split_subset(
        track, {1: 1.9, 2: 2.0, 3: 1.9}, config=config
    )

    assert selected == (1, 3)
