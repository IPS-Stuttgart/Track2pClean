from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bayescatrack.association.multiplane_consistency import (
    PlaneRegistrationQuality,
    apply_multiplane_quality_penalty,
    shared_registration_reliability,
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
