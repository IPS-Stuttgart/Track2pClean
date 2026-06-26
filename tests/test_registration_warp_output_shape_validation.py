import numpy as np
import pytest

from bayescatrack.registration import (
    warp_image_into_reference_frame,
    warp_roi_masks_into_reference_frame,
)


_IDENTITY_MATRIX = np.eye(2)
_ZERO_OFFSET = np.zeros(2)


def test_registration_image_warp_rejects_boolean_output_shape_component():
    image = np.ones((4, 4), dtype=float)

    with pytest.raises(ValueError, match="output_shape"):
        warp_image_into_reference_frame(
            image,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(True, 4),
        )


def test_registration_roi_mask_warp_rejects_fractional_output_shape_component():
    roi_masks = np.ones((1, 4, 4), dtype=float)

    with pytest.raises(ValueError, match="output_shape"):
        warp_roi_masks_into_reference_frame(
            roi_masks,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(4.5, 4),
        )


def test_registration_warps_normalize_integer_like_output_shape_components():
    image = np.arange(16, dtype=float).reshape(4, 4)
    roi_masks = image.reshape(1, 4, 4)

    warped_image = warp_image_into_reference_frame(
        image,
        _IDENTITY_MATRIX,
        _ZERO_OFFSET,
        output_shape=(3.0, np.int64(2)),
    )
    warped_masks = warp_roi_masks_into_reference_frame(
        roi_masks,
        _IDENTITY_MATRIX,
        _ZERO_OFFSET,
        output_shape=(3.0, np.int64(2)),
    )

    assert warped_image.shape == (3, 2)
    assert warped_masks.shape == (1, 3, 2)
