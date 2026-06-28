import numpy as np
import pytest

from bayescatrack.registration import warp_roi_masks_into_reference_frame

_IDENTITY_MATRIX = np.eye(2)
_ZERO_OFFSET = np.zeros(2)


def test_registration_roi_mask_warp_rejects_text_threshold_that_parses_as_float():
    roi_masks = np.ones((1, 4, 4), dtype=float)
    text_threshold = str(1 / 2)

    with pytest.raises(ValueError, match="threshold"):
        warp_roi_masks_into_reference_frame(
            roi_masks,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(4, 4),
            binarize=True,
            threshold=text_threshold,
        )


def test_registration_roi_mask_warp_rejects_text_scalar_threshold_array():
    roi_masks = np.ones((1, 4, 4), dtype=float)
    text_threshold = np.array(str(1 / 2))

    with pytest.raises(ValueError, match="threshold"):
        warp_roi_masks_into_reference_frame(
            roi_masks,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(4, 4),
            binarize=True,
            threshold=text_threshold,
        )


def test_registration_roi_mask_warp_rejects_boolean_scalar_threshold_array():
    roi_masks = np.ones((1, 4, 4), dtype=float)

    with pytest.raises(ValueError, match="threshold"):
        warp_roi_masks_into_reference_frame(
            roi_masks,
            _IDENTITY_MATRIX,
            _ZERO_OFFSET,
            output_shape=(4, 4),
            binarize=True,
            threshold=np.array(True),
        )


def test_registration_roi_mask_warp_accepts_zero_dimensional_numeric_threshold():
    roi_masks = np.ones((1, 4, 4), dtype=float)

    warped_masks = warp_roi_masks_into_reference_frame(
        roi_masks,
        _IDENTITY_MATRIX,
        _ZERO_OFFSET,
        output_shape=(4, 4),
        binarize=True,
        threshold=np.array(1 / 2),
    )

    assert warped_masks.dtype == np.bool_
    assert warped_masks.shape == (1, 4, 4)
