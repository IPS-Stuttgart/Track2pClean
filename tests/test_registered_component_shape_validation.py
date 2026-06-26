"""Regression tests for registered pairwise component shape validation."""

from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.registered_masks import (
    add_registered_roi_validity_components,
    mask_invalid_registered_roi_columns,
)


def _mismatched_pairwise_components() -> dict[str, np.ndarray]:
    return {
        "pairwise_cost_matrix": np.zeros((2, 3), dtype=float),
        "iou": np.ones((2, 2), dtype=float),
    }


def test_invalid_registered_roi_masking_rejects_inconsistent_component_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mask_invalid_registered_roi_columns(
            _mismatched_pairwise_components(),
            valid_registered_rois=np.array([True, True, False]),
        )


def test_registered_roi_validity_components_reject_inconsistent_component_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        add_registered_roi_validity_components(
            _mismatched_pairwise_components(),
            np.array([True, True, False]),
        )
