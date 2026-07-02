from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.core import bridge


def _single_roi_plane() -> bridge.CalciumPlaneData:
    masks = np.zeros((1, 3, 3), dtype=bool)
    masks[0, 1, 1] = True
    return bridge.CalciumPlaneData(masks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weighted_centroids": "false"}, "weighted_centroids must be a boolean"),
        ({"soft_iou": 1}, "soft_iou must be a boolean"),
        ({"return_components": np.asarray([True])}, "return_components must be a boolean"),
    ],
)
def test_core_bridge_rejects_ambiguous_pairwise_boolean_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    plane = _single_roi_plane()

    with pytest.raises(ValueError, match=message):
        plane.build_pairwise_cost_matrix(plane, **kwargs)
