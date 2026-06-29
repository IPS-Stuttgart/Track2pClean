import numpy as np
import pytest
import track2pclean  # noqa: F401
from bayescatrack.fov_registration import apply_subpixel_image_translation


def test_probe():
    image = np.arange(12, dtype=float).reshape(3, 4)
    with pytest.raises(ValueError):
        apply_subpixel_image_translation(image, (0.0, 0.0), output_shape='bad')
