from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.joint_registration_assignment import (
    JointRefinementConfig,
    JointRegistrationAssignmentConfig,
    anchor_relief_cost_matrix,
    high_confidence_anchor_edges,
)


@pytest.mark.parametrize(
    "bad_relief", [True, np.bool_(False), np.nan, np.inf, -np.inf, -0.1]
)
def test_anchor_relief_cost_matrix_rejects_malformed_relief(bad_relief):
    with pytest.raises(ValueError, match="relief"):
        anchor_relief_cost_matrix(
            np.array([[1.0]], dtype=float),
            [(0, 0)],
            relief=bad_relief,
        )


@pytest.mark.parametrize(
    "bad_relief", [True, np.bool_(False), np.nan, np.inf, -np.inf, -0.1]
)
def test_joint_refinement_config_rejects_malformed_cost_relief(bad_relief):
    with pytest.raises(ValueError, match="cost_relief"):
        JointRefinementConfig(cost_relief=bad_relief)


@pytest.mark.parametrize(
    "bad_tolerance", [True, np.bool_(False), np.nan, np.inf, -np.inf, -1.0]
)
def test_joint_refinement_config_rejects_malformed_convergence_tolerance(bad_tolerance):
    with pytest.raises(ValueError, match="convergence_tolerance"):
        JointRefinementConfig(convergence_tolerance=bad_tolerance)


@pytest.mark.parametrize(
    "bad_quantile", [True, np.bool_(False), np.nan, np.inf, -0.01, 1.01]
)
def test_high_confidence_anchor_edges_rejects_malformed_quantile(bad_quantile):
    with pytest.raises(ValueError, match="quantile"):
        high_confidence_anchor_edges(
            np.array([[1.0]], dtype=float),
            quantile=bad_quantile,
        )


@pytest.mark.parametrize("bad_min_anchor_edges", [True, np.bool_(False), 0, -1])
def test_high_confidence_anchor_edges_rejects_malformed_min_anchor_edges(
    bad_min_anchor_edges,
):
    with pytest.raises(ValueError, match="min_anchor_edges"):
        high_confidence_anchor_edges(
            np.array([[1.0]], dtype=float),
            min_anchor_edges=bad_min_anchor_edges,
        )


@pytest.mark.parametrize(
    "bad_margin", [True, np.bool_(False), np.nan, np.inf, -np.inf, -0.1]
)
def test_joint_registration_assignment_config_rejects_malformed_margin(bad_margin):
    with pytest.raises(ValueError, match="min_anchor_margin"):
        JointRegistrationAssignmentConfig(min_anchor_margin=bad_margin)
