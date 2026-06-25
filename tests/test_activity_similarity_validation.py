from __future__ import annotations

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
