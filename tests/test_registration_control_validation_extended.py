from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData, registration


def _single_roi_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2, 2] = True
    return CalciumPlaneData(masks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": np.asarray(["xy"])}, "order must be either 'xy' or 'yx'"),
        (
            {"binarize_registered_masks": "false"},
            "binarize_registered_masks must be a boolean",
        ),
        (
            {"registered_mask_threshold": True},
            "registered_mask_threshold must be a finite scalar in \\[0, 1\\]",
        ),
        (
            {"registered_mask_threshold": 1.5},
            "registered_mask_threshold must be a finite scalar in \\[0, 1\\]",
        ),
    ],
)
def test_register_measurement_plane_rejects_malformed_boundary_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    plane = _single_roi_plane()

    with pytest.raises(ValueError, match=message):
        registration.register_measurement_plane_to_reference(
            plane,
            plane,
            registration_model="translation",
            **kwargs,
        )
