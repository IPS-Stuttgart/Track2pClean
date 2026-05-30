from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from bayescatrack import CalciumPlaneData
from bayescatrack.association.multiplane_consistency import (
    PlaneRegistrationQuality,
    apply_multiplane_quality_penalty,
    shared_registration_reliability,
)
from bayescatrack.association.registered_masks import (
    drop_empty_registered_masks,
    empty_registered_roi_mask,
)
from bayescatrack.association.session_adaptive_calibration import (
    SessionContext,
    context_intercept,
    probability_cost_matrix,
    session_context_from_planes,
)


def _plane(cell_probabilities):
    return SimpleNamespace(
        image_shape=(4, 4),
        n_rois=2,
        cell_probabilities=np.asarray(cell_probabilities, dtype=float),
        traces=None,
        spike_traces=None,
    )


def test_empty_registered_roi_mask_treats_nonfinite_and_nonpositive_masks_as_empty():
    roi_masks = np.asarray(
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.inf, np.inf], [np.inf, np.inf]],
            [[-1.0, -2.0], [0.0, -3.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=float,
    )
    plane = CalciumPlaneData(roi_masks)

    np.testing.assert_array_equal(
        empty_registered_roi_mask(plane),
        np.asarray([True, True, True, False]),
    )


def test_drop_empty_registered_masks_removes_nonfinite_registered_rois():
    roi_masks = np.asarray(
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[np.inf, 0.0], [0.0, np.inf]],
        ],
        dtype=float,
    )
    plane = CalciumPlaneData(
        roi_masks,
        traces=np.arange(6, dtype=float).reshape(3, 2),
        spike_traces=np.arange(6, 12, dtype=float).reshape(3, 2),
        neuropil_traces=np.arange(12, 18, dtype=float).reshape(3, 2),
        cell_probabilities=np.asarray([0.1, 0.9, 0.2]),
        roi_indices=np.asarray([10, 11, 12]),
        roi_features={"radius": np.asarray([1.0, 2.0, 3.0])},
    )

    filtered, empty_mask = drop_empty_registered_masks(plane)

    np.testing.assert_array_equal(empty_mask, np.asarray([True, False, True]))
    assert filtered.roi_masks.shape == (1, 2, 2)
    np.testing.assert_array_equal(filtered.roi_indices, np.asarray([11]))
    np.testing.assert_array_equal(filtered.traces, plane.traces[[1]])
    np.testing.assert_array_equal(filtered.spike_traces, plane.spike_traces[[1]])
    np.testing.assert_array_equal(filtered.neuropil_traces, plane.neuropil_traces[[1]])
    np.testing.assert_allclose(filtered.cell_probabilities, np.asarray([0.9]))
    np.testing.assert_allclose(filtered.roi_features["radius"], np.asarray([2.0]))
    np.testing.assert_allclose(filtered.centroids(order="yx"), np.asarray([[1.0], [1.0]]))


def test_session_context_ignores_all_nonfinite_cell_probabilities():
    context = session_context_from_planes(
        _plane([np.nan, np.inf]),
        _plane([np.nan, -np.inf]),
        session_gap=np.nan,
        registration_metadata={"valid_fraction": np.nan, "fit_rmse": np.inf},
    )

    assert context.session_gap == 1.0
    assert context.mean_cell_probability == 0.5
    assert context.registration_valid_fraction == 1.0
    assert context.registration_fit_rmse == 0.0
    assert np.isfinite(context_intercept(context))


def test_context_intercept_sanitizes_nonfinite_context_fields():
    context = SessionContext(
        session_gap=np.nan,
        roi_density=np.inf,
        mean_cell_probability=np.nan,
        registration_fit_rmse=np.inf,
        registration_valid_fraction=np.nan,
        trace_availability_fraction=np.nan,
        backend_bias=np.nan,
    )

    assert np.isfinite(context_intercept(context))


def test_probability_cost_matrix_treats_nonfinite_probabilities_conservatively():
    costs = probability_cost_matrix(np.asarray([[np.nan, -np.inf, np.inf, 0.5]]))

    assert np.all(np.isfinite(costs))
    assert costs[0, 0] > costs[0, 3]
    assert costs[0, 1] > costs[0, 3]
    assert costs[0, 2] == 0.0


def test_shared_registration_reliability_rejects_all_nonfinite_quality_values():
    reliability = shared_registration_reliability(
        [
            PlaneRegistrationQuality("plane0", np.nan, np.nan),
            PlaneRegistrationQuality("plane1", np.inf, -np.inf),
        ]
    )

    assert reliability == 0.0


def test_multiplane_quality_penalty_remains_finite_for_nonfinite_quality_values():
    adjusted = apply_multiplane_quality_penalty(
        np.zeros((2, 2), dtype=float),
        [
            PlaneRegistrationQuality("plane0", np.nan, np.nan),
            PlaneRegistrationQuality("plane1", np.inf, -np.inf),
        ],
        penalty_weight=2.0,
    )

    np.testing.assert_allclose(adjusted, np.full((2, 2), 2.0))
