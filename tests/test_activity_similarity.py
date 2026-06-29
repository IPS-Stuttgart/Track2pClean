from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.activity_similarity import (
    activity_similarity_components,
    add_activity_similarity_components,
)
from bayescatrack.association.calibrated_costs import pairwise_feature_tensor
from bayescatrack.core.bridge import CalciumPlaneData


def _masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    return masks


def _planes() -> tuple[CalciumPlaneData, CalciumPlaneData]:
    masks = _masks()
    traces = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    reference = CalciumPlaneData(roi_masks=masks, traces=traces, spike_traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces, spike_traces=traces)
    return reference, measurement


@pytest.mark.parametrize(
    "similarity_epsilon",
    [True, np.bool_(True), 0.0, -1.0e-12, np.nan, np.inf, -np.inf],
)
def test_activity_similarity_rejects_invalid_similarity_epsilon(
    similarity_epsilon: object,
) -> None:
    reference, measurement = _planes()

    with pytest.raises(ValueError, match="similarity_epsilon"):
        activity_similarity_components(
            reference,
            measurement,
            similarity_epsilon=similarity_epsilon,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "event_threshold",
    [True, np.bool_(False), np.nan, np.inf, -np.inf],
)
def test_activity_similarity_rejects_invalid_event_threshold(
    event_threshold: object,
) -> None:
    reference, measurement = _planes()

    with pytest.raises(ValueError, match="event_threshold"):
        activity_similarity_components(
            reference,
            measurement,
            event_threshold=event_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "n_rois",
    ["1", b"1", bytearray(b"1"), np.str_("1"), np.bytes_(b"1"), np.array("1")],
)
def test_activity_similarity_rejects_text_like_plane_roi_counts(n_rois: object) -> None:
    reference = SimpleNamespace(n_rois=n_rois)
    measurement = SimpleNamespace(n_rois=1)

    with pytest.raises(ValueError, match="reference_plane.n_rois"):
        activity_similarity_components(reference, measurement)


def test_activity_similarity_add_hook_validates_controls() -> None:
    reference, measurement = _planes()
    pairwise_components = {"pairwise_cost_matrix": np.zeros((2, 2), dtype=float)}

    with pytest.raises(ValueError, match="event_threshold"):
        add_activity_similarity_components(
            pairwise_components,
            reference,
            measurement,
            event_threshold=np.nan,
        )


def test_activity_similarity_favors_matching_trace_structure() -> None:
    masks = _masks()
    traces = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces.copy())

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(np.diag(components["activity_correlation"]), np.ones(2))
    npt.assert_allclose(np.diag(components["activity_similarity"]), np.ones(2))
    npt.assert_allclose(np.diag(components["activity_similarity_cost"]), np.zeros(2))
    assert (
        components["activity_similarity"][0, 0]
        > components["activity_similarity"][0, 1]
    )
    assert (
        components["activity_similarity"][1, 1]
        > components["activity_similarity"][1, 0]
    )
    npt.assert_allclose(components["activity_similarity_available"], np.ones((2, 2)))


def test_activity_similarity_prefers_spike_traces_in_auto_mode() -> None:
    masks = _masks()
    fluorescence = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    spikes = fluorescence[::-1]
    reference = CalciumPlaneData(
        roi_masks=masks, traces=fluorescence, spike_traces=spikes
    )
    measurement = CalciumPlaneData(
        roi_masks=masks, traces=fluorescence, spike_traces=spikes
    )

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(np.diag(components["activity_similarity"]), np.ones(2))
    assert (
        components["activity_similarity"][0, 0]
        > components["activity_similarity"][0, 1]
    )


def test_activity_similarity_uses_pairwise_finite_overlap_for_missing_samples() -> None:
    masks = _masks()
    reference_traces = np.array(
        [
            [0.0, 1.0, np.nan, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, np.nan],
        ]
    )
    measurement_traces = np.array(
        [
            [0.0, 1.0, 0.0, 1.0, np.nan],
            [1.0, 0.0, np.nan, 0.0, 1.0],
        ]
    )
    reference = CalciumPlaneData(roi_masks=masks, traces=reference_traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=measurement_traces)

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(np.diag(components["activity_similarity_available"]), 1.0)
    npt.assert_allclose(np.diag(components["activity_correlation"]), 1.0)
    npt.assert_allclose(np.diag(components["activity_similarity_cost"]), 0.0)
    assert (
        components["activity_similarity"][0, 0]
        > components["activity_similarity"][0, 1]
    )


def test_activity_similarity_hook_is_neutral_when_traces_are_missing() -> None:
    masks = _masks()
    reference = CalciumPlaneData(roi_masks=masks)
    measurement = CalciumPlaneData(roi_masks=masks)

    components = activity_similarity_components(reference, measurement)

    npt.assert_allclose(components["activity_correlation"], np.zeros((2, 2)))
    npt.assert_allclose(components["activity_similarity"], np.zeros((2, 2)))
    npt.assert_allclose(components["activity_similarity_cost"], np.full((2, 2), 0.5))
    npt.assert_allclose(components["activity_similarity_available"], np.zeros((2, 2)))


def test_activity_similarity_components_can_be_added_to_pairwise_component_dict() -> (
    None
):
    masks = _masks()
    traces = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    reference = CalciumPlaneData(roi_masks=masks, traces=traces)
    measurement = CalciumPlaneData(roi_masks=masks, traces=traces)
    pairwise_components = {"pairwise_cost_matrix": np.zeros((2, 2), dtype=float)}

    add_activity_similarity_components(pairwise_components, reference, measurement)
    features = pairwise_feature_tensor(
        pairwise_components,
        feature_names=("activity_similarity_cost", "activity_similarity_available"),
    )

    assert features.shape == (2, 2, 2)
    npt.assert_allclose(
        features[:, :, 0], pairwise_components["activity_similarity_cost"]
    )
    npt.assert_allclose(
        features[:, :, 1], pairwise_components["activity_similarity_available"]
    )


def test_pairwise_feature_tensor_accepts_missing_activity_hook_components() -> None:
    features = pairwise_feature_tensor(
        {"pairwise_cost_matrix": np.zeros((2, 3), dtype=float)},
        feature_names=("activity_similarity_cost", "activity_similarity_available"),
    )

    assert features.shape == (2, 3, 2)
    npt.assert_allclose(features, np.zeros((2, 3, 2)))
