from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.fov_registration import (
    apply_subpixel_image_translation,
    apply_subpixel_roi_mask_translation,
)


@pytest.mark.parametrize(
    "shift_yx",
    [
        [True, 0.0],
        [np.bool_(False), 1.0],
        np.asarray([True, False], dtype=bool),
    ],
)
def test_apply_subpixel_image_translation_rejects_boolean_shift_values(shift_yx):
    with pytest.raises(ValueError, match="shift_yx"):
        apply_subpixel_image_translation(np.zeros((4, 4), dtype=float), shift_yx)


@pytest.mark.parametrize("shift_yx", [[1.0], [1.0, 2.0, 3.0]])
def test_apply_subpixel_image_translation_rejects_malformed_shift_shapes(shift_yx):
    with pytest.raises(ValueError, match="shift_yx"):
        apply_subpixel_image_translation(np.zeros((4, 4), dtype=float), shift_yx)


def test_apply_subpixel_image_translation_preserves_numeric_shifts():
    image = np.zeros((5, 5), dtype=float)
    image[2, 2] = 1.0

    translated = apply_subpixel_image_translation(image, [0.5, -0.5])

    assert translated.shape == image.shape
    assert translated.dtype.kind == "f"
    npt.assert_allclose(np.sum(translated), 1.0)


def test_apply_subpixel_roi_mask_translation_rejects_boolean_shift_values():
    with pytest.raises(ValueError, match="shift_yx"):
        apply_subpixel_roi_mask_translation(
            np.zeros((1, 4, 4), dtype=bool),
            [True, 0.0],
        )
