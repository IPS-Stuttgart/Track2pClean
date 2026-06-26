from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import CalciumPlaneData


def _single_roi_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    return CalciumPlaneData(masks)


def test_position_covariances_rejects_nan_regularization() -> None:
    with pytest.raises(ValueError, match="regularization must be a finite non-negative value"):
        _single_roi_plane().position_covariances(regularization=np.nan)
