from __future__ import annotations

import bayescatrack  # noqa: F401
import numpy as np
import pytest
from bayescatrack.fov_registration import (
    apply_subpixel_image_translation,
    apply_subpixel_roi_mask_translation,
)


@pytest.mark.parametrize(
    "translator,source",
    [
        (apply_subpixel_image_translation, np.zeros((3, 3), dtype=float)),
        (apply_subpixel_roi_mask_translation, np.zeros((1, 3, 3), dtype=bool)),
    ],
)
def test_subpixel_shift_rejects_memoryview_controls(translator, source) -> None:
    with pytest.raises(ValueError, match="shift_yx"):
        translator(source, memoryview(b"\x00\x01"))
