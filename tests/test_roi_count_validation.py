from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.activity_similarity import activity_similarity_components


def _count_plane(n_rois: object) -> SimpleNamespace:
    traces = np.zeros((1, 3), dtype=float)
    return SimpleNamespace(
        n_rois=n_rois,
        traces=traces,
        spike_traces=traces,
        neuropil_traces=None,
    )


@pytest.mark.parametrize(
    ("plane_name", "reference_n_rois", "measurement_n_rois"),
    [
        ("reference_plane", np.array(True), 1),
        ("reference_plane", np.array([1]), 1),
        ("reference_plane", np.array([True]), 1),
        ("measurement_plane", 1, np.array(False)),
        ("measurement_plane", 1, np.array([2])),
        ("measurement_plane", 1, np.array([False])),
    ],
)
def test_activity_similarity_rejects_array_valued_plane_roi_counts(
    plane_name: str,
    reference_n_rois: object,
    measurement_n_rois: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{plane_name}\.n_rois"):
        activity_similarity_components(
            _count_plane(reference_n_rois),
            _count_plane(measurement_n_rois),
        )
