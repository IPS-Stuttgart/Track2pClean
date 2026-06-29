from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.joint_registration_assignment import (
    JointRefinementConfig,
    JointRegistrationAssignmentConfig,
    anchor_relief_cost_matrix,
    high_confidence_anchor_edges,
)


class _BadFloat:
    def __float__(self) -> float:
        raise OverflowError("bad float conversion")


class _BadIndex:
    def __index__(self) -> int:
        raise OverflowError("bad integer conversion")


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


def test_high_confidence_anchor_edges_wraps_overflowing_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        high_confidence_anchor_edges(
            np.array([[1.0]], dtype=float),
            quantile=_BadFloat(),
        )


def test_high_confidence_anchor_edges_wraps_overflowing_min_anchor_edges() -> None:
    with pytest.raises(ValueError, match="min_anchor_edges"):
        high_confidence_anchor_edges(
            np.array([[1.0]], dtype=float),
            min_anchor_edges=_BadIndex(),
        )


def test_anchor_relief_cost_matrix_wraps_overflowing_relief() -> None:
    with pytest.raises(ValueError, match="relief"):
        anchor_relief_cost_matrix(
            np.array([[1.0]], dtype=float),
            [(0, 0)],
            relief=_BadFloat(),
        )


def test_joint_refinement_config_wraps_overflowing_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        JointRefinementConfig(max_iterations=_BadIndex())


def test_joint_refinement_config_wraps_overflowing_high_confidence_quantile() -> None:
    with pytest.raises(ValueError, match="high_confidence_quantile"):
        JointRefinementConfig(high_confidence_quantile=_BadFloat())


def test_joint_refinement_config_wraps_overflowing_cost_relief() -> None:
    with pytest.raises(ValueError, match="cost_relief"):
        JointRefinementConfig(cost_relief=_BadFloat())


def test_joint_registration_config_wraps_overflowing_min_anchor_probability() -> None:
    with pytest.raises(ValueError, match="min_anchor_probability"):
        JointRegistrationAssignmentConfig(min_anchor_probability=_BadFloat())


def test_joint_registration_config_wraps_overflowing_min_anchor_margin() -> None:
    with pytest.raises(ValueError, match="min_anchor_margin"):
        JointRegistrationAssignmentConfig(min_anchor_margin=_BadFloat())


def test_joint_registration_config_wraps_overflowing_min_anchors() -> None:
    with pytest.raises(ValueError, match="min_anchors"):
        JointRegistrationAssignmentConfig(min_anchors=_BadIndex())
