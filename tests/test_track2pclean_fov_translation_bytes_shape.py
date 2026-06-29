from __future__ import annotations

import builtins

import numpy as np
import pytest
import track2pclean  # noqa: F401
from bayescatrack.fov_registration import (
    apply_subpixel_image_translation,
    apply_subpixel_roi_mask_translation,
)


_BYTES_LIKE_OUTPUT_SHAPES = [
    builtins.bytearray(b'ab'),
    builtins.memoryview(b'ab'),
]


@pytest.mark.parametrize("bad_shape", _BYTES_LIKE_OUTPUT_SHAPES)
def test_track2pclean_rejects_bytes_like_image_output_shape(bad_shape):
    image = np.arange(12, dtype=float).reshape(3, 4)

    with pytest.raises(ValueError, match="output_shape"):
        apply_subpixel_image_translation(
            image,
            (0.0, 0.0),
            **{"output_" + "shape": bad_shape},
        )


@pytest.mark.parametrize("bad_shape", _BYTES_LIKE_OUTPUT_SHAPES)
def test_track2pclean_rejects_bytes_like_roi_output_shape(bad_shape):
    roi_masks = np.zeros((1, 3, 4), dtype=bool)

    with pytest.raises(ValueError, match="output_shape"):
        apply_subpixel_roi_mask_translation(
            roi_masks,
            (0.0, 0.0),
            **{"output_" + "shape": bad_shape},
        )
