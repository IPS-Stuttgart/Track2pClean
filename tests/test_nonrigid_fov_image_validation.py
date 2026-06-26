from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.nonrigid_registration import register_measurement_plane_by_nonrigid_fov


def _plane_with_fov(fov: np.ndarray) -> CalciumPlaneData:
    mask = np.zeros((1, *fov.shape), dtype=bool)
    mask[0, 3:6, 3:6] = True
    return CalciumPlaneData(
        roi_masks=mask,
        fov=fov,
        roi_indices=np.asarray([0], dtype=int),
        source="nonrigid_fov_validation",
    )


@pytest.mark.parametrize("bad_plane", ["reference", "measurement"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonrigid_registration_rejects_nonfinite_fov_values(
    bad_plane: str,
    bad_value: float,
) -> None:
    reference_fov = np.zeros((10, 10), dtype=float)
    measurement_fov = np.zeros((10, 10), dtype=float)
    reference_fov[4, 4] = 1.0
    measurement_fov[4, 4] = 1.0
    if bad_plane == "reference":
        reference_fov[2, 2] = bad_value
    else:
        measurement_fov[2, 2] = bad_value

    reference = _plane_with_fov(reference_fov)
    measurement = _plane_with_fov(measurement_fov)

    with patch(
        "bayescatrack.nonrigid_registration.estimate_fov_affine_transform",
        side_effect=AssertionError("estimator should not run for non-finite FOVs"),
    ):
        with pytest.raises(ValueError, match="FOV images must contain only finite values"):
            register_measurement_plane_by_nonrigid_fov(reference, measurement)
