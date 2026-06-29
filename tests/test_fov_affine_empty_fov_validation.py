from __future__ import annotations

import numpy as np
from bayescatrack.fov_affine_registration import estimate_fov_affine_transform

_AFFINE_IDENTITY_XY = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


def test_estimate_fov_affine_transform_returns_identity_for_empty_fovs():
    image = np.zeros((0, 8), dtype=float)

    estimate = estimate_fov_affine_transform(image, image, min_tile_size=2)

    np.testing.assert_allclose(estimate.matrix_xy, _AFFINE_IDENTITY_XY)
    np.testing.assert_allclose(estimate.inverse_matrix_xy, _AFFINE_IDENTITY_XY)
    assert estimate.tile_reference_xy.shape == (0, 2)
    assert estimate.tile_measurement_xy.shape == (0, 2)
    assert estimate.tile_shift_yx.shape == (1, 2)
    assert estimate.fit_rmse == 0.0
    assert estimate.fallback_translation
