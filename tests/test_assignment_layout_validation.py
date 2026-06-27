from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.matching import solve_bundle_linear_assignment


def _bundle(
    *,
    reference_roi_indices=(10, 11),
    measurement_roi_indices=(20, 21),
):
    return SimpleNamespace(
        pairwise_cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        reference_roi_indices=np.asarray(reference_roi_indices),
        measurement_roi_indices=np.asarray(measurement_roi_indices),
        reference_session_name="day0",
        measurement_session_name="day1",
    )


@pytest.mark.parametrize(
    ("reference_roi_indices", "message"),
    [
        ([10], r"bundle\.reference_roi_indices length"),
        ([10, 11, 12], r"bundle\.reference_roi_indices length"),
        (
            np.asarray([[10], [11]]),
            r"bundle\.reference_roi_indices must be one-dimensional",
        ),
    ],
)
def test_solve_bundle_linear_assignment_rejects_reference_roi_layout_mismatch(
    reference_roi_indices,
    message,
):
    with pytest.raises(ValueError, match=message):
        solve_bundle_linear_assignment(
            _bundle(reference_roi_indices=reference_roi_indices),
        )


@pytest.mark.parametrize(
    ("measurement_roi_indices", "message"),
    [
        ([20], r"bundle\.measurement_roi_indices length"),
        ([20, 21, 22], r"bundle\.measurement_roi_indices length"),
        (
            np.asarray([[20], [21]]),
            r"bundle\.measurement_roi_indices must be one-dimensional",
        ),
    ],
)
def test_solve_bundle_linear_assignment_rejects_measurement_roi_layout_mismatch(
    measurement_roi_indices,
    message,
):
    with pytest.raises(ValueError, match=message):
        solve_bundle_linear_assignment(
            _bundle(measurement_roi_indices=measurement_roi_indices),
        )
