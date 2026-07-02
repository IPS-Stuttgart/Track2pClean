from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData
from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    estimate_subpixel_fov_shift,
    register_measurement_plane_by_fov_translation,
)


def _shifted_fov_pair() -> tuple[np.ndarray, np.ndarray]:
    reference_fov = np.zeros((16, 16), dtype=float)
    reference_fov[4:8, 5:9] = 1.0
    measurement_fov = apply_integer_image_translation(reference_fov, np.array([1, -1]))
    return reference_fov, measurement_fov


def test_estimate_subpixel_fov_shift_normalizes_float_conversion_overflow():
    reference_fov, measurement_fov = _shifted_fov_pair()

    with pytest.raises(ValueError, match="refinement_radius"):
        estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            refinement_radius=Fraction(10**400, 1),
        )


@pytest.mark.parametrize(
    "bad_radius",
    [
        "1.0",
        b"1.0",
        np.str_("1.0"),
        np.bytes_(b"1.0"),
        np.array("1.0", dtype=object),
        np.array(b"1.0", dtype=object),
    ],
)
def test_estimate_subpixel_fov_shift_rejects_text_like_refinement_radius(
    bad_radius,
):
    reference_fov, measurement_fov = _shifted_fov_pair()

    with pytest.raises(ValueError, match="refinement_radius"):
        estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            refinement_radius=bad_radius,
        )


def test_register_measurement_plane_normalizes_float_conversion_overflow():
    reference_fov, measurement_fov = _shifted_fov_pair()
    reference_plane = CalciumPlaneData((reference_fov > 0.0)[None, :, :], fov=reference_fov)
    measurement_plane = CalciumPlaneData((measurement_fov > 0.0)[None, :, :], fov=measurement_fov)

    with pytest.raises(ValueError, match="subpixel_refinement_radius"):
        register_measurement_plane_by_fov_translation(
            reference_plane,
            measurement_plane,
            subpixel=True,
            subpixel_refinement_radius=Fraction(10**400, 1),
        )


@pytest.mark.parametrize(
    "bad_radius",
    [
        "1.0",
        b"1.0",
        np.str_("1.0"),
        np.bytes_(b"1.0"),
        np.array("1.0", dtype=object),
        np.array(b"1.0", dtype=object),
    ],
)
def test_register_measurement_plane_rejects_text_like_subpixel_refinement_radius(
    bad_radius,
):
    reference_fov, measurement_fov = _shifted_fov_pair()
    reference_plane = CalciumPlaneData((reference_fov > 0.0)[None, :, :], fov=reference_fov)
    measurement_plane = CalciumPlaneData((measurement_fov > 0.0)[None, :, :], fov=measurement_fov)

    with pytest.raises(ValueError, match="subpixel_refinement_radius"):
        register_measurement_plane_by_fov_translation(
            reference_plane,
            measurement_plane,
            subpixel=True,
            subpixel_refinement_radius=bad_radius,
        )
