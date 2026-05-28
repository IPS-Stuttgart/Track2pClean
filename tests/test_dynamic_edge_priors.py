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


def test_mutual_best_margin_prior_rewards_only_reciprocal_separated_edges() -> None:
    costs = np.asarray(
        [
            [0.5, 2.0, 3.0],
            [2.2, 0.7, 0.8],
            [3.0, 0.9, 0.6],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(
            mutual_best_relief=0.2,
            mutual_best_min_margin=0.5,
            mutual_best_cost_cap=1.0,
        ),
    )

    expected = costs.copy()
    expected[0, 0] -= 0.2
    np.testing.assert_allclose(adjusted, expected)


def test_mutual_best_margin_prior_respects_cost_cap_and_min_cost() -> None:
    costs = np.asarray(
        [
            [0.1, 1.5],
            [1.4, 0.2],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(
            mutual_best_relief=1.0,
            mutual_best_cost_cap=0.15,
            mutual_best_min_cost=0.0,
        ),
    )

    expected = costs.copy()
    expected[0, 0] = 0.0
    np.testing.assert_allclose(adjusted, expected)


def test_mutual_best_margin_prior_does_not_reward_tied_best_edges() -> None:
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
        config=DynamicEdgePriorConfig(mutual_best_relief=0.5),
    )

    np.testing.assert_allclose(adjusted, costs)


def test_dynamic_edge_prior_config_validates_reciprocal_rank_knobs() -> None:
    with pytest.raises(ValueError, match="reciprocal_rank_weight"):
        DynamicEdgePriorConfig(reciprocal_rank_weight=-0.1)
    with pytest.raises(ValueError, match="reciprocal_rank_cap"):
        DynamicEdgePriorConfig(reciprocal_rank_cap=-0.1)


def test_dynamic_edge_prior_config_validates_mutual_best_knobs() -> None:
    with pytest.raises(ValueError, match="mutual_best_relief"):
        DynamicEdgePriorConfig(mutual_best_relief=-0.1)
    with pytest.raises(ValueError, match="mutual_best_min_margin"):
        DynamicEdgePriorConfig(mutual_best_min_margin=-0.1)
    with pytest.raises(ValueError, match="mutual_best_cost_cap"):
        DynamicEdgePriorConfig(mutual_best_cost_cap=-0.1)
    with pytest.raises(ValueError, match="mutual_best_min_cost"):
        DynamicEdgePriorConfig(mutual_best_min_cost=float("nan"))
