from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    apply_integer_roi_mask_translation,
)


@pytest.mark.parametrize(
    "shift_yx",
    [
        np.array([1.2, 0.0]),
        np.array([np.nan, 0.0]),
        np.array([True, 0]),
        np.array([1, 2, 3]),
        np.array([[1], [2]]),
    ],
)
def test_apply_integer_image_translation_rejects_non_integer_shift(shift_yx) -> None:
    with pytest.raises(ValueError, match="shift_yx"):
        apply_integer_image_translation(np.zeros((3, 3)), shift_yx)


def test_apply_integer_roi_mask_translation_validates_shift_for_empty_stack() -> None:
    empty_masks = np.zeros((0, 4, 5), dtype=bool)

    with pytest.raises(ValueError, match="shift_yx"):
        apply_integer_roi_mask_translation(empty_masks, np.array([0.5, 0.0]))


def test_apply_integer_image_translation_accepts_integer_like_shift() -> None:
    image = np.arange(9).reshape(3, 3)

    translated = apply_integer_image_translation(image, np.array([1.0, -1.0]))

    npt.assert_array_equal(translated, np.array([[0, 0, 0], [1, 2, 0], [4, 5, 0]]))
