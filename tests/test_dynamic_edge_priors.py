from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


def test_reciprocal_rank_prior_keeps_mutual_best_edges_unchanged() -> None:
    costs = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(reciprocal_rank_weight=1.0),
    )

    np.testing.assert_allclose(adjusted[0, 0], costs[0, 0])
    np.testing.assert_allclose(adjusted[1, 1], costs[1, 1])
    np.testing.assert_allclose(adjusted[0, 1], costs[0, 1] + np.log(2.0))
    np.testing.assert_allclose(adjusted[1, 0], costs[1, 0] + np.log(2.0))


def test_reciprocal_rank_prior_uses_dense_ranks_for_ties() -> None:
    costs = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(reciprocal_rank_weight=1.0),
    )

    np.testing.assert_allclose(adjusted[0, 0], costs[0, 0])
    np.testing.assert_allclose(adjusted[0, 1], costs[0, 1])
    np.testing.assert_allclose(adjusted[1, 0], costs[1, 0])
    np.testing.assert_allclose(adjusted[1, 1], costs[1, 1] + np.log(2.0))


def test_reciprocal_rank_prior_can_be_capped() -> None:
    costs = np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(
            reciprocal_rank_weight=10.0,
            reciprocal_rank_cap=0.5,
        ),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[1.0, 2.5, 3.5]]))


def test_reciprocal_rank_prior_preserves_gated_large_costs() -> None:
    costs = np.asarray(
        [
            [1.0, 1.0e6],
            [2.0, np.inf],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(reciprocal_rank_weight=1.0, large_cost=1.0e6),
    )

    np.testing.assert_allclose(adjusted[0, 0], costs[0, 0])
    np.testing.assert_allclose(adjusted[1, 0], costs[1, 0] + np.log(2.0))
    assert adjusted[0, 1] == pytest.approx(1.0e6)
    assert adjusted[1, 1] == pytest.approx(1.0e6)


def test_dynamic_edge_prior_config_validates_reciprocal_rank_knobs() -> None:
    with pytest.raises(ValueError, match="reciprocal_rank_weight"):
        DynamicEdgePriorConfig(reciprocal_rank_weight=-0.1)
    with pytest.raises(ValueError, match="reciprocal_rank_cap"):
        DynamicEdgePriorConfig(reciprocal_rank_cap=-0.1)
