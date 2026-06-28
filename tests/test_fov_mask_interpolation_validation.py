from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.fov_registration import (
    apply_image_translation,
    apply_roi_mask_translation,
)


@pytest.mark.parametrize(
    "interpolation",
    [
        np.array(["nearest"], dtype=object),
        ["nearest"],
    ],
)
def test_apply_image_translation_rejects_array_like_interpolation_controls(
    interpolation,
):
    image = np.zeros((3, 3), dtype=float)

    with pytest.raises(ValueError, match="mask_interpolation"):
        apply_image_translation(image, (0.0, 0.0), interpolation=interpolation)


@pytest.mark.parametrize(
    "interpolation",
    [
        np.array(["bilinear"], dtype=object),
        ["bilinear"],
    ],
)
def test_apply_roi_mask_translation_rejects_array_like_interpolation_controls(
    interpolation,
):
    roi_masks = np.zeros((1, 3, 3), dtype=float)

    with pytest.raises(ValueError, match="mask_interpolation"):
        apply_roi_mask_translation(roi_masks, (0.0, 0.0), interpolation=interpolation)
