from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association import dynamic_edge_priors as dynamic_edge_priors_module
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)


def test_empty_registered_roi_mask_rejects_string_truthiness() -> None:
    with pytest.raises(ValueError, match="empty_registered_rois"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0, 2.0]], dtype=float),
            {},
            session_gap=1,
            empty_registered_rois=["False", "True"],
            config=DynamicEdgePriorConfig(registration_empty_roi_weight=5.0),
        )


def test_empty_registered_roi_mask_rejects_non_binary_numeric_values() -> None:
    with pytest.raises(ValueError, match="empty_registered_rois"):
        apply_dynamic_edge_priors(
            np.asarray([[1.0, 2.0]], dtype=float),
            {},
            session_gap=1,
            empty_registered_rois=[0, 2],
            config=DynamicEdgePriorConfig(registration_empty_roi_weight=5.0),
        )


def test_empty_registered_roi_mask_still_accepts_binary_numeric_masks() -> None:
    adjusted = apply_dynamic_edge_priors(
        np.asarray([[1.0, 2.0]], dtype=float),
        {},
        session_gap=1,
        empty_registered_rois=[0, 1],
        config=DynamicEdgePriorConfig(registration_empty_roi_weight=5.0),
    )

    np.testing.assert_allclose(adjusted, np.asarray([[1.0, 7.0]], dtype=float))


def test_empty_registered_roi_core_mask_helper_rejects_string_truthiness() -> None:
    with pytest.raises(ValueError, match="empty_registered_rois"):
        dynamic_edge_priors_module._column_mask_for_cost_shape(["False", "True"], (1, 2))


def test_empty_registered_roi_core_mask_helper_rejects_non_binary_numeric_values() -> None:
    with pytest.raises(ValueError, match="empty_registered_rois"):
        dynamic_edge_priors_module._column_mask_for_cost_shape([0, 2], (1, 2))
