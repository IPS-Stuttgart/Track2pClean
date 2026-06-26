from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import CalciumPlaneData
from bayescatrack.association.shifted_overlap import (
    install_shifted_overlap_cost_patch,
    shifted_iou_pairwise_cost_matrix,
)


def _planes() -> tuple[CalciumPlaneData, CalciumPlaneData]:
    reference = np.zeros((1, 6, 6), dtype=bool)
    measurement = np.zeros((1, 6, 6), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 3:5] = True
    return CalciumPlaneData(reference), CalciumPlaneData(measurement)


def test_shifted_overlap_rejects_ambiguous_boolean_knobs() -> None:
    reference_plane, measurement_plane = _planes()
    original_method = install_shifted_overlap_cost_patch()
    try:
        with pytest.raises(
            ValueError,
            match="use_shifted_iou_for_iou_cost must be a boolean",
        ):
            reference_plane.build_pairwise_cost_matrix(
                measurement_plane,
                shifted_iou_radius=1,
                use_shifted_iou_for_iou_cost="false",
            )
        with pytest.raises(
            ValueError,
            match="use_shifted_mask_cosine_for_mask_cosine_cost must be a boolean",
        ):
            reference_plane.build_pairwise_cost_matrix(
                measurement_plane,
                shifted_iou_radius=1,
                use_shifted_mask_cosine_for_mask_cosine_cost=1,
            )
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_method  # type: ignore[method-assign]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"shifted_iou_weight": np.nan},
            "shifted_iou_weight must be a finite non-negative value",
        ),
        (
            {"shifted_mask_cosine_weight": np.inf},
            "shifted_mask_cosine_weight must be a finite non-negative value",
        ),
        (
            {"shifted_iou_shift_penalty_weight": True},
            "shifted_iou_shift_penalty_weight must be a finite non-negative value",
        ),
        (
            {
                "shifted_iou_shift_penalty_weight": 1.0,
                "shifted_iou_shift_penalty_scale": np.nan,
            },
            "shifted_iou_shift_penalty_scale must be a finite positive value",
        ),
    ],
)
def test_shifted_overlap_rejects_nonfinite_or_boolean_float_knobs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    reference_plane, measurement_plane = _planes()
    original_method = install_shifted_overlap_cost_patch()
    try:
        with pytest.raises(ValueError, match=message):
            reference_plane.build_pairwise_cost_matrix(
                measurement_plane,
                shifted_iou_radius=1,
                **kwargs,
            )
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_method  # type: ignore[method-assign]


def test_shifted_overlap_direct_wrapper_uses_same_scalar_validation() -> None:
    reference_plane, measurement_plane = _planes()

    def original_method(
        self: CalciumPlaneData,
        other: CalciumPlaneData,
        **_kwargs: object,
    ) -> np.ndarray:
        assert self is reference_plane
        assert other is measurement_plane
        return np.zeros((1, 1), dtype=float)

    with pytest.raises(
        ValueError,
        match="use_shifted_iou_for_iou_cost must be a boolean",
    ):
        shifted_iou_pairwise_cost_matrix(
            original_method,
            reference_plane,
            measurement_plane,
            shifted_iou_radius=1,
            use_shifted_iou_for_iou_cost="false",
        )
