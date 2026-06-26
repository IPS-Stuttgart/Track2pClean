from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.nonrigid_registration import register_measurement_plane_by_nonrigid_fov
from bayescatrack.track2p_registration import register_plane_pair


def _plane() -> CalciumPlaneData:
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 3:6, 3:6] = True
    fov = np.zeros((10, 10), dtype=float)
    fov[4, 4] = 1.0
    return CalciumPlaneData(
        roi_masks=mask,
        fov=fov,
        roi_indices=np.asarray([0], dtype=int),
        source="control_validation",
    )


def _identity_estimate() -> SimpleNamespace:
    return SimpleNamespace(
        inverse_matrix_xy=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        tile_reference_xy=np.empty((0, 2), dtype=float),
        tile_measurement_xy=np.empty((0, 2), dtype=float),
        tile_peak_correlation=np.empty(0, dtype=float),
        fit_rmse=0.0,
        fallback_translation=False,
    )


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("tps_regularization", True),
        ("tps_regularization", np.nan),
        ("tps_regularization", -1.0e-3),
        ("bspline_regularization", np.bool_(False)),
        ("bspline_regularization", np.inf),
        ("bspline_regularization", -1.0e-3),
        ("optical_flow_alpha", True),
        ("optical_flow_alpha", np.nan),
        ("optical_flow_alpha", 0.0),
        ("optical_flow_alpha", -1.0),
        ("optical_flow_iterations", True),
        ("optical_flow_iterations", np.array([1])),
        ("optical_flow_iterations", 1.5),
        ("optical_flow_iterations", -1),
    ],
)
def test_nonrigid_registration_rejects_invalid_runtime_controls(
    kwarg: str,
    value: object,
) -> None:
    reference = _plane()
    measurement = _plane()

    with patch(
        "bayescatrack.nonrigid_registration.estimate_fov_affine_transform",
        side_effect=AssertionError("estimator should not run for invalid controls"),
    ):
        with pytest.raises(ValueError, match=kwarg):
            register_measurement_plane_by_nonrigid_fov(
                reference,
                measurement,
                **{kwarg: value},
            )


def test_nonrigid_registration_accepts_valid_numpy_scalar_controls() -> None:
    reference = _plane()
    measurement = _plane()

    with patch(
        "bayescatrack.nonrigid_registration.estimate_fov_affine_transform",
        return_value=_identity_estimate(),
    ) as estimator:
        registration = register_measurement_plane_by_nonrigid_fov(
            reference,
            measurement,
            tps_regularization=np.asarray(0.0),
            bspline_regularization=np.float64(0.01),
            optical_flow_iterations=np.int64(0),
            optical_flow_alpha=np.float64(25.0),
        )

    estimator.assert_called_once()
    assert registration.transform_type == "bspline"
    assert registration.registered_measurement_plane.ops is not None
    assert (
        registration.registered_measurement_plane.ops[
            "nonrigid_registration_optical_flow_iterations"
        ]
        == 0
    )


def test_nonrigid_registration_public_pair_route_uses_control_validation() -> None:
    reference = _plane()
    measurement = _plane()

    with patch(
        "bayescatrack.nonrigid_registration.estimate_fov_affine_transform",
        side_effect=AssertionError("estimator should not run for invalid controls"),
    ):
        with pytest.raises(ValueError, match="optical_flow_alpha"):
            register_plane_pair(
                reference,
                measurement,
                transform_type="bspline",
                registration_options={"optical_flow_alpha": np.nan},
            )
