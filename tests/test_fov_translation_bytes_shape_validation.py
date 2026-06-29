from __future__ import annotations

import numpy as np
import pytest

import track2pclean  # noqa: F401  # installs the compatibility validation patch
from bayescatrack import fov_registration

_OUTPUT_SHAPE_ERROR = "output_shape must contain exactly two positive integer dimensions"


@pytest.mark.parametrize(
    "output_shape",
    [
        b"12",
        np.bytes_(b"12"),
        bytearray(b"12"),
        memoryview(b"12"),
    ],
)
def test_fov_translation_rejects_bytes_like_output_shapes(output_shape):
    image = np.ones((2, 2), dtype=float)
    masks = np.ones((1, 2, 2), dtype=bool)
    calls = (
        (fov_registration.apply_integer_image_translation, (image, (0, 0))),
        (fov_registration.apply_subpixel_image_translation, (image, (0.0, 0.0))),
        (fov_registration.apply_integer_roi_mask_translation, (masks, (0, 0))),
        (fov_registration.apply_subpixel_roi_mask_translation, (masks, (0.0, 0.0))),
    )

    for function, args in calls:
        with pytest.raises(ValueError, match=_OUTPUT_SHAPE_ERROR):
            function(*args, output_shape=output_shape)
