from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.joint_registration_assignment import (
    JointRefinementConfig,
    JointRegistrationAssignmentConfig,
    anchor_relief_cost_matrix,
    high_confidence_anchor_edges,
)


@pytest.mark.parametrize("array_value", [np.array(0.10), np.array([0.10])])
def test_high_confidence_anchor_edges_rejects_array_quantile(array_value):
    with pytest.raises(ValueError, match="quantile"):
        high_confidence_anchor_edges(
            np.array([[1.0]], dtype=float), quantile=array_value
        )


@pytest.mark.parametrize("array_value", [np.array(1), np.array([1])])
def test_high_confidence_anchor_edges_rejects_array_min_anchor_edges(array_value):
    with pytest.raises(ValueError, match="min_anchor_edges"):
        high_confidence_anchor_edges(
            np.array([[1.0]], dtype=float),
            min_anchor_edges=array_value,
        )


@pytest.mark.parametrize("array_value", [np.array(0.25), np.array([0.25])])
def test_anchor_relief_cost_matrix_rejects_array_relief(array_value):
    with pytest.raises(ValueError, match="relief"):
        anchor_relief_cost_matrix(
            np.array([[1.0]], dtype=float),
            [(0, 0)],
            relief=array_value,
        )


@pytest.mark.parametrize("array_value", [np.array(3), np.array([3])])
def test_joint_refinement_config_rejects_array_max_iterations(array_value):
    with pytest.raises(ValueError, match="max_iterations"):
        JointRefinementConfig(max_iterations=array_value)


@pytest.mark.parametrize("array_value", [np.array(0.90), np.array([0.90])])
def test_joint_registration_assignment_config_rejects_array_probability(array_value):
    with pytest.raises(ValueError, match="min_anchor_probability"):
        JointRegistrationAssignmentConfig(min_anchor_probability=array_value)
