from __future__ import annotations

import builtins

import numpy as np
import pytest
import track2pclean  # noqa: F401
from bayescatrack.fov_registration import apply_subpixel_image_translation


def test_track2pclean_rejects_bytes_like_image_output_shape():
    image = np.arange(12, dtype=float).reshape(3, 4)
    bad_shape = builtins.memoryview(b'ab')

    with pytest.raises(ValueError, match="output_shape"):
        apply_subpixel_image_translation(
            image,
            (0.0, 0.0),
            **{"output_" + "shape": bad_shape},
        )
