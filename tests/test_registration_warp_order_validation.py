import numpy as np
import pytest
from bayescatrack.registration import (
    warp_image_into_reference_frame,
    warp_roi_masks_into_reference_frame,
)

_IDENTITY_MATRIX = np.eye(2)
_ZERO_OFFSET = np.zeros(2)


def test_registration_image_warp_rejects_array_like_coordinate_order():
    image = np.ones((4, 4), dtype=float)

    with pytest.raises(ValueError, match="order"):
        warp_image_into_reference_frame(
            image,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(4, 4),
            order=["xy"],
        )


def test_registration_roi_mask_warp_rejects_array_like_coordinate_order():
    roi_masks = np.ones((1, 4, 4), dtype=float)

    with pytest.raises(ValueError, match="order"):
        warp_roi_masks_into_reference_frame(
            roi_masks,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(4, 4),
            order=np.asarray(["xy"], dtype=object),
        )
