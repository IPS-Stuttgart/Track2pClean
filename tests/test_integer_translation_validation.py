from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    apply_integer_roi_mask_translation,
)


def test_apply_integer_image_translation_accepts_integer_like_float_shift():
    image = np.arange(9).reshape(3, 3)

    translated = apply_integer_image_translation(
        image,
        np.array([1.0, -1.0]),
        fill_value=-1,
    )

    npt.assert_array_equal(
        translated,
        np.array([[-1, -1, -1], [1, 2, -1], [4, 5, -1]]),
    )


@pytest.mark.parametrize(
    "shift_yx",
    [
        np.array([1.5, 0.0]),
        np.array([np.nan, 0.0]),
        [True, 0],
        np.array([1, 0, 2]),
    ],
)
def test_apply_integer_image_translation_rejects_malformed_shift(shift_yx):
    image = np.arange(9).reshape(3, 3)

    with pytest.raises(ValueError, match="shift_yx"):
        apply_integer_image_translation(image, shift_yx)


def test_apply_integer_roi_mask_translation_rejects_fractional_shift():
    roi_masks = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 1] = True

    with pytest.raises(ValueError, match="shift_yx"):
        apply_integer_roi_mask_translation(roi_masks, np.array([0.5, 0.0]))
