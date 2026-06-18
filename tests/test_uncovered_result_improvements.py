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
    AdaptiveCalibrationConfig,
    SessionAdaptiveCalibrationConfig,
    SessionContext,
    apply_context_intercept_to_costs,
    apply_session_context_offset,
    probability_cost_matrix,
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
    ("field", "value"),
    [
        ("min_overlap_fraction", True),
        ("min_overlap_fraction", float("nan")),
        ("min_overlap_fraction", float("inf")),
        ("min_overlap_fraction", -0.1),
        ("min_overlap_fraction", 1.1),
        ("min_weighted_dice", False),
        ("min_weighted_dice", float("nan")),
        ("min_weighted_dice", float("inf")),
        ("min_weighted_dice", -0.1),
        ("min_weighted_dice", 1.1),
        ("max_area_ratio_cost", True),
        ("max_area_ratio_cost", float("nan")),
        ("max_area_ratio_cost", float("inf")),
        ("max_area_ratio_cost", -0.1),
        ("min_children", False),
        ("min_children", 1.5),
        ("min_children", 1),
        ("max_children", True),
        ("max_children", 1.5),
        ("max_children", 1),
    ],
)
def test_segmentation_event_config_rejects_invalid_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        SegmentationEventConfig(**{field: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("min_similarity", True),
        ("min_similarity", float("nan")),
        ("min_similarity", float("inf")),
        ("min_similarity", -0.1),
        ("min_similarity", 1.1),
        ("min_children", False),
        ("min_children", 1.5),
        ("min_children", 1),
        ("max_children", True),
        ("max_children", 1.5),
        ("max_children", 1),
    ],
)
def test_detect_segmentation_events_rejects_invalid_mapping_controls(
    key: str, value: float | bool
) -> None:
    components = {"weighted_dice_similarity": np.asarray([[0.9]], dtype=float)}

    with pytest.raises(ValueError, match=key):
        detect_segmentation_events(components, config={key: value})


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


def test_multi_hypothesis_edge_set_consensus_respects_min_votes() -> None:
    candidates_a = ((0, 1, 0, 1), (0, 1, 1, 2))
    candidates_b = ((0, 1, 0, 1), (0, 1, 1, 3))

    consensus = consensus_edges((candidates_a, candidates_b), min_votes=2)

    assert consensus == {(0, 1, 0, 1): 2}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("edge_top_k", True),
        ("edge_top_k", 1.5),
        ("edge_top_k", 0),
        ("beam_width", False),
        ("beam_width", 1.5),
        ("beam_width", 0),
        ("min_consensus_votes", True),
        ("min_consensus_votes", 1.5),
        ("min_consensus_votes", 0),
        ("max_edge_cost", True),
        ("max_edge_cost", float("nan")),
        ("max_edge_cost", float("inf")),
        ("fill_value", True),
        ("fill_value", 1.5),
    ],
)
def test_hypothesis_config_rejects_invalid_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        HypothesisConfig(**{field: value})


@pytest.mark.parametrize(
    "row_top_k",
    [True, False, 1.5, 0],
)
def test_top_k_edge_candidates_rejects_invalid_row_top_k(
    row_top_k: float | bool,
) -> None:
    with pytest.raises(ValueError, match="row_top_k"):
        top_k_edge_candidates([[1.0]], edge=(0, 1), row_top_k=row_top_k)


@pytest.mark.parametrize("max_cost", [True, float("nan"), float("inf")])
def test_top_k_edge_candidates_rejects_invalid_max_cost(
    max_cost: float | bool,
) -> None:
    with pytest.raises(ValueError, match="max_cost"):
        top_k_edge_candidates([[1.0]], edge=(0, 1), max_cost=max_cost)


@pytest.mark.parametrize("min_votes", [True, False, 1.5, 0])
def test_consensus_edges_rejects_invalid_min_votes(
    min_votes: float | bool,
) -> None:
    with pytest.raises(ValueError, match="min_votes"):
        consensus_edges((((0, 1, 0, 1),),), min_votes=min_votes)


@pytest.mark.parametrize(
    "min_support_fraction",
    [True, False, 0.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_consensus_edges_rejects_invalid_support_fraction(
    min_support_fraction: float | bool,
) -> None:
    with pytest.raises(ValueError, match="min_support_fraction"):
        consensus_edges(
            (((0, 1, 0, 1),),), min_support_fraction=min_support_fraction
        )


@pytest.mark.parametrize("fill_value", [True, 1.5])
def test_consensus_edges_rejects_invalid_fill_value(
    fill_value: float | bool,
) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        consensus_edges((((0, 1, 0, 1),),), fill_value=fill_value)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_intercept", True),
        ("base_intercept", float("nan")),
        ("base_intercept", float("inf")),
        ("session_gap_weight", False),
        ("session_gap_weight", float("nan")),
        ("session_gap_weight", float("inf")),
        ("roi_density_weight", True),
        ("roi_density_weight", float("nan")),
        ("roi_density_weight", float("inf")),
        ("low_cell_probability_weight", False),
        ("low_cell_probability_weight", float("nan")),
        ("low_cell_probability_weight", float("inf")),
        ("registration_rmse_weight", True),
        ("registration_rmse_weight", float("nan")),
        ("registration_rmse_weight", float("inf")),
        ("invalid_warp_weight", False),
        ("invalid_warp_weight", float("nan")),
        ("invalid_warp_weight", float("inf")),
        ("trace_available_weight", True),
        ("trace_available_weight", float("nan")),
        ("trace_available_weight", float("inf")),
        ("max_abs_intercept", False),
        ("max_abs_intercept", float("nan")),
        ("max_abs_intercept", float("inf")),
        ("max_abs_intercept", 0.0),
    ],
)
def test_adaptive_calibration_config_rejects_invalid_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        AdaptiveCalibrationConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_gap_weight", True),
        ("session_gap_weight", float("nan")),
        ("session_gap_weight", float("inf")),
        ("session_gap_weight", -0.1),
        ("registration_rmse_weight", False),
        ("registration_rmse_weight", float("nan")),
        ("registration_rmse_weight", float("inf")),
        ("registration_rmse_weight", -0.1),
        ("invalid_fraction_weight", True),
        ("invalid_fraction_weight", float("nan")),
        ("invalid_fraction_weight", float("inf")),
        ("invalid_fraction_weight", -0.1),
        ("low_cell_probability_weight", False),
        ("low_cell_probability_weight", float("nan")),
        ("low_cell_probability_weight", float("inf")),
        ("low_cell_probability_weight", -0.1),
    ],
)
def test_session_adaptive_calibration_config_rejects_invalid_controls(
    field: str, value: float | bool
) -> None:
    with pytest.raises(ValueError, match=field):
        SessionAdaptiveCalibrationConfig(**{field: value})


@pytest.mark.parametrize("epsilon", [True, float("nan"), float("inf"), 0.0])
def test_probability_cost_matrix_rejects_invalid_epsilon(
    epsilon: float | bool,
) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        probability_cost_matrix([[0.5]], epsilon=epsilon)


@pytest.mark.parametrize("temperature", [False, float("nan"), float("inf"), 0.0])
def test_apply_context_intercept_to_costs_rejects_invalid_temperature(
    temperature: float | bool,
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        apply_context_intercept_to_costs(
            [[1.0]], SessionContext(), temperature=temperature
        )


@pytest.mark.parametrize("offset", [True, float("nan"), float("inf")])
def test_apply_session_context_offset_rejects_invalid_offset(
    offset: float | bool,
) -> None:
    with pytest.raises(ValueError, match="offset"):
        apply_session_context_offset([[1.0]], offset)
