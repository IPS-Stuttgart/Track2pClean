from __future__ import annotations

import numpy as np
import numpy.testing as npt

from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    apply_integer_roi_mask_translation,
)


def test_apply_integer_image_translation_handles_shifts_outside_output():
    image = np.arange(30, dtype=int).reshape(10, 3)

    shifted = apply_integer_image_translation(
        image, np.array([6, 0]), output_shape=(5, 3), fill_value=-1
    )

    npt.assert_array_equal(shifted, np.full((5, 3), -1, dtype=int))


def test_apply_integer_roi_mask_translation_handles_shifts_outside_output():
    masks = np.ones((2, 10, 3), dtype=bool)

    translated = apply_integer_roi_mask_translation(masks, [6, 0], output_shape=(5, 3))

    assert not np.any(translated)
