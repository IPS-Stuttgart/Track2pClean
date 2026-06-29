from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.fov_registration import apply_integer_image_translation


class _OverflowingIndex:
    def __index__(self) -> int:
        raise OverflowError("index too large")


def test_integer_image_translation_rejects_overflowing_index_component() -> None:
    image = np.zeros((3, 3), dtype=float)

    with pytest.raises(
        ValueError,
        match="shift_yx must contain exactly two integer values",
    ):
        apply_integer_image_translation(image, (_OverflowingIndex(), 0))
