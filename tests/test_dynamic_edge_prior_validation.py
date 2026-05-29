from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


def test_dynamic_edge_prior_rejects_nonfinite_edge_quality_bias():
    for invalid_bias in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="edge_quality_bias must be finite"):
            DynamicEdgePriorConfig(edge_quality_bias=invalid_bias)


def test_dynamic_edge_prior_allows_finite_negative_edge_quality_bias():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=1,
        config=DynamicEdgePriorConfig(edge_quality_bias=-0.5),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[2.5, 3.5]], dtype=float))


def test_dynamic_edge_prior_rejects_invalid_session_gap_when_gap_weighted():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    for invalid_gap in (0, -1, True, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="session_gap must be a finite value"):
            apply_dynamic_edge_priors(
                costs,
                {},
                session_gap=invalid_gap,
                config=DynamicEdgePriorConfig(session_gap_weight=0.25),
            )


def test_dynamic_edge_prior_session_gap_validation_keeps_valid_offsets():
    costs = np.asarray([[3.0, 4.0]], dtype=float)

    adjusted = apply_dynamic_edge_priors(
        costs,
        {},
        session_gap=3,
        config=DynamicEdgePriorConfig(session_gap_weight=0.25),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[3.5, 4.5]], dtype=float))
