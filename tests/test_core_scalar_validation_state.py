from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import CalciumPlaneData


def _single_roi_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    return CalciumPlaneData(masks)


def test_position_covariances_rejects_boolean_positional_regularization() -> None:
    with pytest.raises(ValueError, match="regularization must be a finite non-negative value"):
        _single_roi_plane().position_covariances("xy", False, True)


def test_constant_velocity_state_moments_rejects_boolean_velocity_variance() -> None:
    with pytest.raises(ValueError, match="velocity_variance must be a finite non-negative value"):
        _single_roi_plane().to_constant_velocity_state_moments(velocity_variance=True)


def test_constant_velocity_state_moments_rejects_nan_regularization() -> None:
    with pytest.raises(ValueError, match="regularization must be a finite non-negative value"):
        _single_roi_plane().to_constant_velocity_state_moments(regularization=np.nan)
