"""Unit tests for result-improvement modules not covered by open PRs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bayescatrack.association.context_descriptors import (
    fov_patch_moments,
    local_density_descriptor,
    pairwise_context_components,
)
from bayescatrack.association.growth_priors import (
    affine_growth_residuals,
    estimate_affine_growth_field,
    growth_penalty_matrix,
)
from bayescatrack.association.joint_registration_assignment import (
    JointRegistrationAssignmentConfig,
    high_confidence_anchor_pairs,
)
from bayescatrack.association.multi_hypothesis import (
    consensus_edges,
    top_k_edge_candidates,
)
from bayescatrack.association.multiplane_consistency import (
    PlaneRegistrationQuality,
    apply_multiplane_quality_penalty,
    shared_registration_reliability,
)
from bayescatrack.association.segmentation_events import detect_segmentation_events
from bayescatrack.association.session_adaptive_calibration import (
    SessionAdaptiveCalibrationConfig,
    apply_session_context_offset,
    session_context_cost_offset,
)


@dataclass
class DummyPlane:
    n_rois: int
    image_shape: tuple[int, int] = (100, 100)
    cell_probabilities: np.ndarray | None = None
    traces: np.ndarray | None = None
    spike_traces: np.ndarray | None = None


def test_segmentation_events_detect_split_and_merge() -> None:
    components = {
        "weighted_dice_similarity": np.asarray(
            [[0.8, 0.7, 0.1], [0.9, 0.85, 0.05]], dtype=float
        )
    }

    events = detect_segmentation_events(components, config={"min_similarity": 0.6})

    event_types = {event.event_type for event in events}
    assert "split" in event_types
    assert "merge" in event_types
    assert any(
        event.measurement_positions == (0, 1)
        for event in events
        if event.event_type == "split"
    )


def test_context_descriptors_return_pairwise_planes_and_patch_moments() -> None:
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [20.0, 20.0]])
    measurement = np.asarray([[0.0, 0.0], [20.0, 21.0]])

    density = local_density_descriptor(reference, radius=2.0)
    components = pairwise_context_components(
        reference, measurement, config={"density_radius": 2.0}
    )
    moments = fov_patch_moments(
        np.arange(25, dtype=float).reshape(5, 5), [[2.0, 2.0]], patch_radius=1
    )

    assert density.tolist() == [1.0, 1.0, 0.0]
    assert components["local_density_cost"].shape == (3, 2)
    assert moments.shape == (1, 2)
    assert np.isclose(moments[0, 0], 12.0)


def test_multi_hypothesis_candidates_and_consensus() -> None:
    candidates_a = top_k_edge_candidates(
        [[1.0, 0.2, 3.0], [0.5, 2.0, 0.1]], edge=(0, 1), row_top_k=1
    )
    candidates_b = top_k_edge_candidates(
        [[1.0, 0.1, 3.0], [0.5, 2.0, 0.1]], edge=(0, 1), row_top_k=1
    )

    consensus = consensus_edges((candidates_a, candidates_b), min_support_fraction=1.0)

    assert (0, 1, 0, 1) in consensus
    assert (0, 1, 1, 2) in consensus


def test_joint_registration_anchor_selection_uses_probability_and_margin() -> None:
    probabilities = np.asarray([[0.95, 0.03], [0.55, 0.45], [0.02, 0.97]])

    anchors = high_confidence_anchor_pairs(
        probabilities,
        config=JointRegistrationAssignmentConfig(
            min_anchor_probability=0.9,
            min_anchor_margin=0.2,
            min_anchors=1,
        ),
    )

    assert anchors.tolist() == [[0, 0], [2, 1]]


def test_growth_priors_prefer_affine_consistent_matches() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = source + np.asarray([2.0, 3.0])

    affine = estimate_affine_growth_field(source, target)
    residuals = affine_growth_residuals(source, target, affine=affine)
    penalties = growth_penalty_matrix(source, target, affine=affine)

    assert np.max(residuals) < 1.0e-10
    assert np.argmin(penalties, axis=1).tolist() == [0, 1, 2]


def test_multiplane_quality_penalty_and_session_offset() -> None:
    qualities = (
        PlaneRegistrationQuality("plane0", registration_rmse=0.0, valid_fraction=1.0),
        PlaneRegistrationQuality("plane1", registration_rmse=1.0, valid_fraction=0.5),
    )
    reliability = shared_registration_reliability(qualities)
    penalized = apply_multiplane_quality_penalty(
        np.zeros((2, 2)), qualities, penalty_weight=2.0
    )

    offset = session_context_cost_offset(
        DummyPlane(10, cell_probabilities=np.asarray([0.5, 1.0])),
        DummyPlane(20, cell_probabilities=np.asarray([0.25, 0.75])),
        session_gap=3,
        registration_metadata={"fit_rmse": 2.0, "valid_fraction": 0.75},
        config=SessionAdaptiveCalibrationConfig(
            session_gap_weight=0.5,
            registration_rmse_weight=0.25,
            invalid_fraction_weight=1.0,
            low_cell_probability_weight=0.1,
        ),
    )
    shifted = apply_session_context_offset(np.zeros((1, 1)), offset)

    assert 0.0 < reliability < 1.0
    assert np.all(penalized > 0.0)
    assert shifted[0, 0] > 0.0
