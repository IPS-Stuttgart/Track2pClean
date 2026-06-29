from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.context_descriptors import fov_patch_moment_descriptors


@pytest.mark.parametrize(
    "image",
    [
        np.array([[2.0]], dtype=float),
        np.array([[1.0, 2.0, 3.0]], dtype=float),
        np.array([[1.0], [2.0], [3.0]], dtype=float),
    ],
)
def test_fov_patch_moment_descriptors_accepts_degenerate_fov_axes(image: np.ndarray):
    descriptors = fov_patch_moment_descriptors(
        image,
        np.array([[0.0, 0.0]], dtype=float),
        patch_radius=2,
        histogram_bins=4,
    )

    assert descriptors.shape == (1, 10)
    assert np.all(np.isfinite(descriptors))
    assert descriptors[0, 4] == 0.0
    assert descriptors[0, 5] >= 1.0
