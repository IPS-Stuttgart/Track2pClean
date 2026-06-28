from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData
from bayescatrack.registration_selection import select_registration_transform


def _single_roi_plane(source: str = "plane") -> CalciumPlaneData:
    roi_masks = np.zeros((1, 5, 5), dtype=bool)
    roi_masks[0, 2:4, 2:4] = True
    fov = np.arange(25, dtype=float).reshape(5, 5)
    return CalciumPlaneData(roi_masks=roi_masks, fov=fov, source=source)


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("numeric adapter overflowed")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"min_fov_correlation_gain": np.nan},
            "min_fov_correlation_gain must be a finite non-negative value",
        ),
        (
            {"min_fov_correlation_gain": True},
            "min_fov_correlation_gain must be a finite non-negative value",
        ),
        (
            {"min_fov_correlation_gain": "0.05"},
            "min_fov_correlation_gain must be a finite non-negative value",
        ),
        (
            {"min_fov_correlation_gain": np.asarray([0.05])},
            "min_fov_correlation_gain must be a finite non-negative value",
        ),
        (
            {"min_fov_correlation_gain": _OverflowingFloat()},
            "min_fov_correlation_gain must be a finite non-negative value",
        ),
        (
            {"max_empty_roi_fraction": np.inf},
            r"max_empty_roi_fraction must be a finite value in \[0, 1\]",
        ),
        (
            {"max_empty_roi_fraction": -0.1},
            r"max_empty_roi_fraction must be a finite value in \[0, 1\]",
        ),
        (
            {"min_retained_mask_area_fraction": np.nan},
            "min_retained_mask_area_fraction must be a finite non-negative value",
        ),
        (
            {"min_nonrigid_inverse_warp_valid_fraction": 1.1},
            r"min_nonrigid_inverse_warp_valid_fraction must be a finite value in \[0, 1\]",
        ),
        (
            {"empty_roi_penalty": np.inf},
            "empty_roi_penalty must be a finite non-negative value",
        ),
        (
            {"retained_area_penalty": True},
            "retained_area_penalty must be a finite non-negative value",
        ),
        (
            {"nonrigid_valid_fraction_penalty": np.nan},
            "nonrigid_valid_fraction_penalty must be a finite non-negative value",
        ),
        (
            {"complexity_penalty": {"none": np.nan}},
            r"complexity_penalty\['none'\] must be a finite non-negative value",
        ),
        (
            {"complexity_penalty": {"none": True}},
            r"complexity_penalty\['none'\] must be a finite non-negative value",
        ),
        (
            {"complexity_penalty": {"none": "0.0"}},
            r"complexity_penalty\['none'\] must be a finite non-negative value",
        ),
        (
            {"complexity_penalty": {"none": np.asarray([0.0])}},
            r"complexity_penalty\['none'\] must be a finite non-negative value",
        ),
        (
            {"complexity_penalty": {"none": _OverflowingFloat()}},
            r"complexity_penalty\['none'\] must be a finite non-negative value",
        ),
    ],
)
def test_auto_registration_selector_rejects_malformed_control_scalars(
    kwargs: dict[str, object], message: str
) -> None:
    reference = _single_roi_plane("reference")
    moving = _single_roi_plane("moving")

    with pytest.raises(ValueError, match=message):
        select_registration_transform(
            reference,
            moving,
            candidate_transforms=("none",),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("candidate_transforms", "message"),
    [
        (("none", "fov-tranlsation"), "unknown transform type"),
        (("none", 1), "candidate_transforms must contain transform-type strings"),
        (("none", "auto"), "'auto' must not be nested inside auto-registration candidates"),
    ],
)
def test_auto_registration_selector_rejects_malformed_candidate_transforms(
    candidate_transforms: object,
    message: str,
) -> None:
    reference = _single_roi_plane("reference")
    moving = _single_roi_plane("moving")

    with pytest.raises(ValueError, match=message):
        select_registration_transform(
            reference,
            moving,
            candidate_transforms=candidate_transforms,
        )


@pytest.mark.parametrize(
    ("complexity_penalty", "message"),
    [
        ({"fov-tranlsation": 0.0}, "complexity_penalty contains unknown transform type"),
        ({1: 0.0}, "complexity_penalty keys must be transform-type strings"),
        ({"auto": 0.0}, "'auto' must not have a complexity penalty"),
        ({"none": 0.0, " none ": 0.1}, "duplicate transform type 'none'"),
    ],
)
def test_auto_registration_selector_rejects_malformed_complexity_penalty_keys(
    complexity_penalty: object,
    message: str,
) -> None:
    reference = _single_roi_plane("reference")
    moving = _single_roi_plane("moving")

    with pytest.raises(ValueError, match=message):
        select_registration_transform(
            reference,
            moving,
            candidate_transforms=("none",),
            complexity_penalty=complexity_penalty,
        )


def test_auto_registration_selector_keeps_finite_scores_for_valid_controls() -> None:
    reference = _single_roi_plane("reference")
    moving = _single_roi_plane("moving")

    result = select_registration_transform(
        reference,
        moving,
        candidate_transforms=("none",),
        min_fov_correlation_gain=0.0,
        max_empty_roi_fraction=1.0,
        min_retained_mask_area_fraction=0.0,
        min_nonrigid_inverse_warp_valid_fraction=0.0,
        empty_roi_penalty=0.0,
        retained_area_penalty=0.0,
        nonrigid_valid_fraction_penalty=0.0,
        complexity_penalty={"none": 0.0},
    )

    assert result.selected_transform_type == "none"
    assert np.isfinite(result.selected_diagnostics.score)
    assert np.isfinite(result.registered_plane.ops["registration_auto_score"])


def test_auto_registration_selector_accepts_single_string_candidate() -> None:
    reference = _single_roi_plane("reference")
    moving = _single_roi_plane("moving")

    result = select_registration_transform(
        reference,
        moving,
        candidate_transforms="none",
    )

    assert result.selected_transform_type == "none"
    assert tuple(candidate.transform_type for candidate in result.diagnostics) == ("none",)


def test_auto_registration_selector_normalizes_complexity_penalty_keys() -> None:
    reference = _single_roi_plane("reference")
    moving = _single_roi_plane("moving")

    result = select_registration_transform(
        reference,
        moving,
        candidate_transforms="none",
        complexity_penalty={" none ": 0.0},
    )

    assert result.selected_transform_type == "none"
    assert tuple(candidate.transform_type for candidate in result.diagnostics) == ("none",)
