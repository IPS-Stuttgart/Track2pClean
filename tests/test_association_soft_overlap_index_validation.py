from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.soft_overlap import dilate_mask_stack


class BrokenIntegerLike:
    def __index__(self) -> int:
        raise ValueError("bad integer conversion")

    def __float__(self) -> float:
        return 1.0


def test_dilate_mask_stack_rejects_broken_integer_like_radius() -> None:
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 1, 1] = True

    with pytest.raises(ValueError, match="radius must be an integer"):
        dilate_mask_stack(mask, radius=BrokenIntegerLike())  # type: ignore[arg-type]
