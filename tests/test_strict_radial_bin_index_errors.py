from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.advanced_roi_components import mask_shape_descriptors


class _OverflowIndex:
    def __index__(self) -> int:
        raise OverflowError("synthetic radial-bin overflow")


class _ValueIndex:
    def __index__(self) -> int:
        raise ValueError("synthetic radial-bin value error")


@pytest.mark.parametrize("bad_value", [_OverflowIndex(), _ValueIndex()])
def test_shape_descriptors_normalize_bad_radial_bin_index_protocol_errors(
    bad_value: object,
) -> None:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True

    with pytest.raises(ValueError, match="radial_bins must be an integer"):
        mask_shape_descriptors(masks, radial_bins=bad_value)  # type: ignore[arg-type]
