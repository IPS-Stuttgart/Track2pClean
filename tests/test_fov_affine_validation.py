from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.fov_affine_registration import (
    apply_affine_image_warp,
    apply_affine_roi_mask_warp,
)

_AFFINE_IDENTITY_XY = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


def test_apply_affine_image_warp_rejects_nonfinite_matrix_before_all_fill_output():
    image = np.ones((4, 4), dtype=float)
    bad_matrix = _AFFINE_IDENTITY_XY.copy()
    bad_matrix[0, 2] = np.nan

    with pytest.raises(ValueError, match="finite 2-by-3 affine matrix"):
        apply_affine_image_warp(image, bad_matrix, output_shape=(4, 4))


def test_apply_affine_roi_mask_warp_rejects_nonfinite_bilinear_matrix():
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1, 1] = True
    bad_matrix = _AFFINE_IDENTITY_XY.copy()
    bad_matrix[1, 2] = np.inf

    with pytest.raises(ValueError, match="finite 2-by-3 affine matrix"):
        apply_affine_roi_mask_warp(
            masks,
            bad_matrix,
            output_shape=(4, 4),
            mode="bilinear",
        )


@pytest.mark.parametrize(
    "bad_output_shape",
    [
        (4.5, 4),
        (True, 4),
        (-1, 4),
        (4,),
    ],
)
def test_apply_affine_image_warp_rejects_malformed_output_shape(bad_output_shape):
    image = np.ones((4, 4), dtype=float)

    with pytest.raises(ValueError, match="output_shape"):
        apply_affine_image_warp(
            image,
            _AFFINE_IDENTITY_XY,
            output_shape=bad_output_shape,
        )


@pytest.mark.parametrize(
    "bad_output_shape",
    [
        (4.5, 4),
        (np.bool_(True), 4),
        (-1, 4),
        (4,),
    ],
)
def test_apply_affine_roi_mask_warp_rejects_malformed_output_shape(bad_output_shape):
    masks = np.zeros((1, 4, 4), dtype=bool)

    with pytest.raises(ValueError, match="output_shape"):
        apply_affine_roi_mask_warp(
            masks,
            _AFFINE_IDENTITY_XY,
            output_shape=bad_output_shape,
        )
