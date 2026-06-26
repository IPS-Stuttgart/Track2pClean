from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.association.activity_similarity import activity_similarity_components
from bayescatrack.core.bridge import CalciumPlaneData


def _activity_plane() -> CalciumPlaneData:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    traces = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    return CalciumPlaneData(roi_masks=masks, traces=traces, spike_traces=traces)


def _count_plane(n_rois: object) -> SimpleNamespace:
    traces = np.zeros((1, 3), dtype=float)
    return SimpleNamespace(
        n_rois=n_rois,
        traces=traces,
        spike_traces=traces,
        neuropil_traces=None,
    )


@pytest.mark.parametrize(
    "similarity_epsilon",
    [0.0, -1.0, np.nan, np.inf, -np.inf, True, np.array(True), np.array([1.0])],
)
def test_activity_similarity_rejects_invalid_similarity_epsilon(
    similarity_epsilon: object,
) -> None:
    plane = _activity_plane()

    with pytest.raises(ValueError, match="similarity_epsilon"):
        activity_similarity_components(
            plane,
            plane,
            similarity_epsilon=similarity_epsilon,
        )


@pytest.mark.parametrize(
    "event_threshold",
    [np.nan, np.inf, -np.inf, False, np.array(True), np.array([0.0])],
)
def test_activity_similarity_rejects_invalid_event_threshold(
    event_threshold: object,
) -> None:
    plane = _activity_plane()

    with pytest.raises(ValueError, match="event_threshold"):
        activity_similarity_components(
            plane,
            plane,
            event_threshold=event_threshold,
        )


@pytest.mark.parametrize(
    ("plane_name", "reference_n_rois", "measurement_n_rois"),
    [
        ("reference_plane", True, 1),
        ("reference_plane", 1.5, 1),
        ("reference_plane", -1, 1),
        ("reference_plane", float("nan"), 1),
        ("measurement_plane", 1, False),
        ("measurement_plane", 1, 2.5),
        ("measurement_plane", 1, -1),
        ("measurement_plane", 1, float("inf")),
    ],
)
def test_activity_similarity_rejects_invalid_plane_roi_counts(
    plane_name: str,
    reference_n_rois: object,
    measurement_n_rois: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{plane_name}\.n_rois"):
        activity_similarity_components(
            _count_plane(reference_n_rois),
            _count_plane(measurement_n_rois),
        )
