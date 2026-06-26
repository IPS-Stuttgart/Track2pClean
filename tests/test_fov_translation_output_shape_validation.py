from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    apply_integer_roi_mask_translation,
    apply_subpixel_image_translation,
    apply_subpixel_roi_mask_translation,
)


_IMAGE_TRANSLATION_FUNCTIONS = [
    apply_integer_image_translation,
    apply_subpixel_image_translation,
]

_ROI_TRANSLATION_FUNCTIONS = [
    apply_integer_roi_mask_translation,
    apply_subpixel_roi_mask_translation,
]


@pytest.mark.parametrize("translation_function", _IMAGE_TRANSLATION_FUNCTIONS)
@pytest.mark.parametrize(
    "bad_output_shape",
    [
        (3.5, 4),
        (np.nan, 4),
        (True, 4),
        (0, 4),
        (-1, 4),
        (4,),
        [[4, 4]],
        ("4", 4),
    ],
)
def test_image_translation_rejects_malformed_output_shape(
    translation_function, bad_output_shape
):
    image = np.arange(12, dtype=float).reshape(3, 4)

    with pytest.raises(ValueError, match="output_shape"):
        translation_function(
            image,
            (0, 0),
            output_shape=bad_output_shape,
        )


@pytest.mark.parametrize("translation_function", _ROI_TRANSLATION_FUNCTIONS)
@pytest.mark.parametrize(
    "bad_output_shape",
    [
        (3.5, 4),
        (np.inf, 4),
        (np.bool_(True), 4),
        (3, 0),
        (3, -1),
        (4,),
        [[4, 4]],
        ("4", 4),
    ],
)
def test_roi_translation_rejects_malformed_output_shape(
    translation_function, bad_output_shape
):
    roi_masks = np.zeros((1, 3, 4), dtype=bool)

    with pytest.raises(ValueError, match="output_shape"):
        translation_function(
            roi_masks,
            (0, 0),
            output_shape=bad_output_shape,
        )


def test_translation_output_shape_accepts_integer_like_dimensions():
    image = np.arange(12, dtype=float).reshape(3, 4)

    translated = apply_subpixel_image_translation(
        image,
        (0.0, 0.0),
        output_shape=(np.int64(3), 4.0),
    )

    assert translated.shape == (3, 4)
