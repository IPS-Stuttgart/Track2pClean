from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.matching import build_track_rows_from_bundles


def _association_bundle(
    cost_shape,
    reference_roi_indices,
    measurement_roi_indices,
    reference_session_name,
    measurement_session_name,
):
    return SimpleNamespace(
        pairwise_cost_matrix=np.zeros(cost_shape, dtype=float),
        reference_roi_indices=np.asarray(reference_roi_indices),
        measurement_roi_indices=np.asarray(measurement_roi_indices),
        reference_session_name=reference_session_name,
        measurement_session_name=measurement_session_name,
    )


def test_build_track_rows_from_bundles_rejects_inconsistent_intermediate_roi_layout():
    bundles = [
        _association_bundle((1, 2), [0], [10, 11], "day0", "day1"),
        _association_bundle((1, 1), [10], [20], "day1", "day2"),
    ]

    with pytest.raises(
        ValueError,
        match="consecutive bundles disagree on ROI indices",
    ):
        build_track_rows_from_bundles(bundles, start_session_index=1)
