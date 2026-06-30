from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.advanced_uncertainty import (
    edge_reliability_matrix,
    edge_uncertainty_config_from_mapping,
    uncertainty_aware_cost_matrix,
)


def test_uncertainty_aware_cost_matrix_accepts_mapping_config() -> None:
    result = uncertainty_aware_cost_matrix(
        np.asarray([[0.0, 1.0]], dtype=float),
        config={"temperature": 1.0, "uncertainty_penalty_weight": 0.0},
    )

    np.testing.assert_allclose(result.adjusted_cost_matrix, np.asarray([[0.0, 1.0]], dtype=float))
    assert result.posterior_probability_matrix.shape == (1, 2)


def test_edge_reliability_matrix_accepts_mapping_config() -> None:
    reliability = edge_reliability_matrix(
        (1, 2),
        {"gated": np.asarray([[1.0, 0.0]], dtype=float)},
        config={"gated_edge_weight": 10.0, "min_reliability": 0.25},
    )

    np.testing.assert_allclose(reliability, np.asarray([[0.25, 1.0]], dtype=float))


def test_edge_uncertainty_config_rejects_non_mapping_config() -> None:
    with pytest.raises(ValueError, match="EdgeUncertaintyConfig"):
        edge_uncertainty_config_from_mapping("temperature=1.0")
