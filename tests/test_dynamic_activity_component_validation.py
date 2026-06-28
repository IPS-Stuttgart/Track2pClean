from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


def test_activity_missing_prior_rejects_wrong_shape_availability_component() -> None:
    costs = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="activity_similarity_available.*wrong shape"):
        apply_dynamic_edge_priors(
            costs,
            {"activity_similarity_available": np.ones((2, 1), dtype=float)},
            session_gap=1,
            config=DynamicEdgePriorConfig(activity_missing_weight=1.0),
        )


def test_activity_missing_prior_rejects_wrong_shape_instead_of_falling_back() -> None:
    costs = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )
    components = {
        "activity_tiebreaker_missing": np.ones((1, 2), dtype=float),
        "activity_similarity_available": np.zeros_like(costs),
    }

    with pytest.raises(ValueError, match="activity_tiebreaker_missing.*wrong shape"):
        apply_dynamic_edge_priors(
            costs,
            components,
            session_gap=1,
            config=DynamicEdgePriorConfig(activity_missing_weight=1.0),
        )
