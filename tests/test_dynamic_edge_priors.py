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


def test_dynamic_edge_prior_bias_does_not_resurrect_invalid_edges() -> None:
    costs = np.asarray(
        [
            [1.0, 1.0e6],
            [np.inf, 2.0e6],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(edge_quality_bias=-5.0, large_cost=1.0e6),
    )

    assert adjusted[0, 0] == pytest.approx(-4.0)
    assert adjusted[0, 1] == pytest.approx(1.0e6)
    assert adjusted[1, 0] == pytest.approx(1.0e6)
    assert adjusted[1, 1] == pytest.approx(1.0e6)


def test_component_dynamic_priors_preserve_invalid_edges() -> None:
    costs = np.asarray(
        [
            [1.0, 1.0e6],
            [np.inf, 2.0],
        ],
        dtype=float,
    )
    components = {
        "cell_probability_cost": np.ones_like(costs),
        "area_ratio_cost": 2.0 * np.ones_like(costs),
        "activity_similarity_available": np.zeros_like(costs),
    }

    adjusted = apply_dynamic_edge_priors(
        costs,
        components,
        session_gap=3,
        empty_registered_rois=np.asarray([False, True]),
        config=DynamicEdgePriorConfig(
            session_gap_weight=0.25,
            cell_probability_weight=1.0,
            area_ratio_weight=0.5,
            activity_missing_weight=2.0,
            registration_empty_roi_weight=8.0,
            large_cost=1.0e6,
        ),
    )

    assert adjusted[0, 0] == pytest.approx(1.0 + 0.5 + 1.0 + 1.0 + 2.0)
    assert adjusted[1, 1] == pytest.approx(2.0 + 0.5 + 1.0 + 1.0 + 2.0 + 8.0)
    assert adjusted[0, 1] == pytest.approx(1.0e6)
    assert adjusted[1, 0] == pytest.approx(1.0e6)


def test_local_margin_prior_penalizes_ambiguous_best_edges() -> None:
    costs = np.asarray(
        [
            [1.0, 1.2],
            [1.3, 0.5],
        ],
        dtype=float,
    )

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(
            local_margin_weight=2.0,
            local_margin_target=0.5,
        ),
    )

    # (0, 0) is locally best but only separated by 0.2 along its row.
    assert adjusted[0, 0] == pytest.approx(1.0 + 2.0 * 0.3)
    # (1, 1) is clearly separated by at least 0.7 and receives no penalty.
    assert adjusted[1, 1] == pytest.approx(0.5)
    # Non-best alternatives are handled by reciprocal-rank penalties, not margins.
    assert adjusted[0, 1] == pytest.approx(1.2)
    assert adjusted[1, 0] == pytest.approx(1.3)


def test_local_margin_prior_can_be_capped_and_handles_ties() -> None:
    costs = np.asarray([[1.0, 1.0, 2.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(
            local_margin_weight=10.0,
            local_margin_target=1.0,
            local_margin_cap=0.25,
        ),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[1.25, 1.25, 2.0]]))


def test_local_margin_prior_preserves_gated_large_costs() -> None:
    costs = np.asarray([[1.0, 1.0e6], [1.3, np.inf]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(
            local_margin_weight=2.0,
            local_margin_target=0.5,
            large_cost=1.0e6,
        ),
    )

    assert adjusted[0, 0] == pytest.approx(1.0 + 2.0 * 0.2)
    assert adjusted[0, 1] == pytest.approx(1.0e6)
    assert adjusted[1, 1] == pytest.approx(1.0e6)


def test_dynamic_edge_prior_config_validates_reciprocal_rank_knobs() -> None:
    with pytest.raises(ValueError, match="reciprocal_rank_weight"):
        DynamicEdgePriorConfig(reciprocal_rank_weight=-0.1)
    with pytest.raises(ValueError, match="reciprocal_rank_cap"):
        DynamicEdgePriorConfig(reciprocal_rank_cap=-0.1)


def test_dynamic_edge_prior_config_validates_edge_quality_bias() -> None:
    with pytest.raises(ValueError, match="edge_quality_bias"):
        DynamicEdgePriorConfig(edge_quality_bias=float("nan"))


def test_dynamic_edge_prior_config_validates_local_margin_knobs() -> None:
    with pytest.raises(ValueError, match="local_margin_weight"):
        DynamicEdgePriorConfig(local_margin_weight=-0.1)
    with pytest.raises(ValueError, match="local_margin_target"):
        DynamicEdgePriorConfig(local_margin_target=float("nan"))
    with pytest.raises(ValueError, match="local_margin_cap"):
        DynamicEdgePriorConfig(local_margin_cap=-0.1)
