"""Regression tests for registered-mask control validation."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.registered_masks import (
    add_registered_roi_validity_components,
    expand_registered_pairwise_cost_columns,
    expand_registered_roi_columns,
    mask_invalid_registered_roi_columns,
)


def _pairwise_components() -> dict[str, np.ndarray]:
    return {
        "pairwise_cost_matrix": np.array([[0.0, 1.0]]),
        "iou": np.array([[0.5, 0.25]]),
        "gated": np.array([[False, False]]),
    }


@pytest.mark.parametrize(
    "valid_registered_rois",
    [
        ["false", "true"],
        [1, 0],
        np.array([np.nan, 0.0]),
        np.array([True, "false"], dtype=object),
    ],
)
def test_registered_roi_validity_rejects_non_boolean_vectors(
    valid_registered_rois: object,
) -> None:
    with pytest.raises(
        ValueError, match="valid_registered_rois must be a boolean vector"
    ):
        mask_invalid_registered_roi_columns(
            _pairwise_components(),
            valid_registered_rois=valid_registered_rois,
        )


def test_registered_roi_validity_masks_invalid_columns_explicitly() -> None:
    masked = mask_invalid_registered_roi_columns(
        _pairwise_components(),
        valid_registered_rois=np.array([True, False]),
        large_cost=7.0,
    )

    assert masked["pairwise_cost_matrix"].tolist() == [[0.0, 7.0]]
    assert masked["iou"].tolist() == [[0.5, 0.0]]
    assert masked["gated"].tolist() == [[False, True]]
    assert masked["registered_roi_valid"].tolist() == [[True, False]]
    assert masked["registered_roi_invalid_cost"].tolist() == [[0.0, 7.0]]


def test_registered_roi_validity_accepts_original_full_length_mask_for_compact_components() -> (
    None
):
    masked = mask_invalid_registered_roi_columns(
        _pairwise_components(),
        valid_registered_rois=np.array([True, False, True]),
        large_cost=7.0,
    )

    assert masked["pairwise_cost_matrix"].tolist() == [[0.0, 1.0]]
    assert masked["registered_roi_valid"].tolist() == [[True, True]]
    assert masked["registered_roi_invalid_cost"].tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize(
    "empty_registered_rois",
    [
        ["false", "true"],
        [1, 0],
        np.array([np.nan, 0.0]),
        np.array([False, "true"], dtype=object),
    ],
)
def test_empty_registered_roi_mask_rejects_non_boolean_vectors(
    empty_registered_rois: object,
) -> None:
    with pytest.raises(
        ValueError, match="empty_registered_rois must be a boolean vector"
    ):
        expand_registered_roi_columns(
            np.array([[1.0]]),
            empty_registered_rois,
            fill_value=np.nan,
        )


@pytest.mark.parametrize("large_cost", [True, False, 0.0, -1.0, np.nan, np.inf, "bad"])
def test_registered_mask_large_cost_rejects_invalid_values(large_cost: object) -> None:
    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        mask_invalid_registered_roi_columns(
            _pairwise_components(),
            valid_registered_rois=np.array([True, False]),
            large_cost=large_cost,
        )

    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        add_registered_roi_validity_components(
            _pairwise_components(),
            np.array([True, False]),
            large_cost=large_cost,
        )

    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        expand_registered_pairwise_cost_columns(
            np.array([[1.0]]),
            np.array([False]),
            large_cost=large_cost,
        )
