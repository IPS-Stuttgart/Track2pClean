import numpy as np
import pytest

from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


def test_reciprocal_rank_prior_rewards_confident_mutual_edges():
    costs = np.array(
        [
            [0.1, 2.0],
            [1.8, 0.2],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config={
            "reciprocal_rank_relief": 0.5,
            "reciprocal_rank_min_margin": 0.5,
        },
    )

    assert adjusted[0, 0] == pytest.approx(-0.4)
    assert adjusted[1, 1] == pytest.approx(-0.3)
    assert adjusted[0, 1] == pytest.approx(costs[0, 1])
    assert adjusted[1, 0] == pytest.approx(costs[1, 0])


def test_reciprocal_rank_prior_penalizes_one_sided_top_candidates():
    costs = np.array(
        [
            [0.10, 0.11],
            [0.09, 2.00],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config={
            "reciprocal_rank_penalty": 0.3,
            "reciprocal_rank_relief": 0.0,
        },
    )

    assert adjusted[0, 0] == pytest.approx(0.40)
    assert adjusted[0, 1] == pytest.approx(0.41)
    assert adjusted[1, 0] == pytest.approx(0.09)
    assert adjusted[1, 1] == pytest.approx(2.00)


def test_reciprocal_rank_prior_respects_large_cost_sentinel():
    costs = np.array(
        [
            [0.1, 1.0e6],
            [2.0, 0.2],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config={
            "reciprocal_rank_relief": 0.5,
            "reciprocal_rank_penalty": 0.5,
            "large_cost": 1.0e6,
        },
    )

    assert adjusted[0, 1] == pytest.approx(1.0e6)


def test_reciprocal_rank_prior_can_be_limited_to_consecutive_edges():
    costs = np.array([[0.1, 2.0], [1.8, 0.2]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=2,
        config=DynamicEdgePriorConfig(
            reciprocal_rank_relief=0.5,
            reciprocal_rank_consecutive_only=True,
        ),
    )

    np.testing.assert_allclose(adjusted, costs)


def test_reciprocal_rank_config_rejects_nonpositive_rank():
    with pytest.raises(ValueError, match="reciprocal_rank_max_rank"):
        DynamicEdgePriorConfig(reciprocal_rank_max_rank=0)
