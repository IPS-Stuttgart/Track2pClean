"""Unit tests for result-improvement modules not covered by open PRs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
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
    HypothesisConfig,
    consensus_edges,
    edge_union_costs,
    top_k_edge_candidates,
)
from bayescatrack.association.multiplane_consistency import (
    PlaneRegistrationQuality,
    apply_multiplane_quality_penalty,
    shared_registration_reliability,
)
from bayescatrack.association.segmentation_events import (
    SegmentationEventConfig,
    detect_segmentation_events,
)
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


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: SegmentationEventConfig(max_area_ratio_cost=np.nan),
            "max_area_ratio_cost must be finite",
        ),
        (
            lambda: SegmentationEventConfig(min_children=1),
            "min_children must be at least two",
        ),
        (
            lambda: SegmentationEventConfig(min_children=2.5),
            "min_children must be a positive integer",
        ),
        (
            lambda: SegmentationEventConfig(max_children=1),
            "max_children must be >= min_children",
        ),
        (
            lambda: detect_segmentation_events(
                {"weighted_dice_similarity": np.ones((2, 2))},
                config={"min_similarity": np.nan},
            ),
            "min_similarity must be finite",
        ),
        (
            lambda: detect_segmentation_events(
                {"weighted_dice_similarity": np.ones((2, 2))},
                config={"min_children": 1.5},
            ),
            "min_children must be a positive integer",
        ),
    ],
)
def test_segmentation_events_reject_silent_candidate_knob_coercions(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


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


def test_context_descriptors_validate_centroid_matrix_shape() -> None:
    assert local_density_descriptor([], radius=2.0).shape == (0,)

    bad_centroids = np.zeros((2, 3), dtype=float)
    message = r"centroids_xy must have shape \(n_roi, 2\)"
    with pytest.raises(ValueError, match=message):
        local_density_descriptor(bad_centroids, radius=2.0)
    with pytest.raises(ValueError, match=message):
        pairwise_context_components(bad_centroids, np.zeros((1, 2), dtype=float))
    with pytest.raises(ValueError, match=message):
        fov_patch_moments(np.ones((4, 4), dtype=float), bad_centroids, patch_radius=1)


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


def test_multi_hypothesis_edge_set_consensus_respects_min_votes() -> None:
    candidates_a = ((0, 1, 0, 1), (0, 1, 1, 2))
    candidates_b = ((0, 1, 0, 1), (0, 1, 1, 3))

    consensus = consensus_edges((candidates_a, candidates_b), min_votes=2)

    assert consensus == {(0, 1, 0, 1): 2}


def test_multi_hypothesis_four_session_python_track_matrix_is_not_edge_set() -> None:
    tracks_a = [[10, 11, 12, 13], [20, -1, 22, 23]]
    tracks_b = [[10, 11, 12, 13]]

    consensus = consensus_edges((tracks_a, tracks_b), min_votes=2)

    assert consensus == {
        (0, 1, 10, 11): 2,
        (1, 2, 11, 12): 2,
        (2, 3, 12, 13): 2,
    }


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: top_k_edge_candidates([[1.0]], edge=(1, 0)),
            "point forward",
        ),
        (
            lambda: consensus_edges((((1, 0, 0, 1),),), min_votes=1),
            "point forward",
        ),
        (
            lambda: edge_union_costs(({(1, 0, 0, 1): 1},)),
            "point forward",
        ),
        (
            lambda: edge_union_costs(({(0, 1, -1, 1): 1},)),
            "edge source_roi must be a non-negative integer",
        ),
        (
            lambda: edge_union_costs(({(0, 1, 0): 1},)),
            "four-item consensus edge",
        ),
    ],
)
def test_multi_hypothesis_rejects_malformed_consensus_edges(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: HypothesisConfig(edge_top_k=1.5),
            "edge_top_k must be a positive integer",
        ),
        (
            lambda: HypothesisConfig(max_edge_cost=np.nan),
            "max_edge_cost must be finite",
        ),
        (
            lambda: top_k_edge_candidates([[1.0]], edge=(0, 1), row_top_k=True),
            "row_top_k must be finite",
        ),
        (
            lambda: top_k_edge_candidates([[1.0]], edge=(0, 1), max_cost=np.nan),
            "max_cost must be finite",
        ),
        (
            lambda: consensus_edges((((0, 1, 0, 1),),), min_votes=0),
            "min_votes must be a positive integer",
        ),
        (
            lambda: consensus_edges((((0, 1, 0, 1),),), min_support_fraction=0.0),
            r"min_support_fraction must be a finite value in \(0, 1\)",
        ),
        (
            lambda: edge_union_costs(({(0, 1, 0, 1): 0},)),
            "vote_count must be a positive integer",
        ),
    ],
)
def test_multi_hypothesis_rejects_silent_candidate_knob_coercions(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


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
    residuals = affine_growth_residuals(source, target, affine)
    costs = growth_penalty_matrix(source, target, affine=affine, scale=1.0)

    assert np.all(residuals < 1e-8)
    assert costs[0, 0] < costs[0, 1]


def test_multiplane_registration_penalty_combines_inverse_qualities() -> None:
    costs = np.zeros((2, 2), dtype=float)
    qualities = {
        0: PlaneRegistrationQuality(plane_name="plane0", fov_correlation=0.5, residual_median=1.0),
        1: PlaneRegistrationQuality(plane_name="plane1", fov_correlation=1.0, residual_median=0.0),
    }

    adjusted = apply_multiplane_quality_penalty(costs, qualities, row_planes=[0, 1], col_planes=[1, 0], weight=2.0)

    assert adjusted[0, 0] > 0.0
    assert adjusted[1, 0] < adjusted[0, 0]
    assert shared_registration_reliability(qualities, [0, 1]) < 1.0


def test_session_adaptive_calibration_offsets_costs() -> None:
    base = np.ones((2, 2), dtype=float)
    context = {"motion_energy_delta": 2.0, "session_gap": 3.0}
    offset = session_context_cost_offset(
        context,
        config=SessionAdaptiveCalibrationConfig(
            motion_energy_weight=0.5,
            session_gap_weight=0.25,
        ),
    )
    adjusted = apply_session_context_offset(base, context, config={"motion_energy_weight": 0.5})

    assert np.isclose(offset, 1.5)
    assert np.allclose(adjusted, base + 1.0)
