from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    ReferenceTrainingOptions,
    supervised_pairwise_mask_from_reference,
)
from bayescatrack.reference import Track2pReference


def test_supervised_pairwise_mask_from_reference_uses_only_annotated_endpoints():
    reference = Track2pReference(
        session_names=("s0", "s1"),
        suite2p_indices=np.array(
            [
                [0, 10],
                [2, None],
                [None, 12],
            ],
            dtype=object,
        ),
    )

    supervised = supervised_pairwise_mask_from_reference(
        reference,
        0,
        1,
        reference_roi_indices=np.array([0, 1, 2], dtype=int),
        measurement_roi_indices=np.array([10, 11, 12], dtype=int),
    )

    expected = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(supervised, expected)


def test_calibrated_association_model_normalizes_direct_string_feature_names():
    model = CalibratedAssociationModel(
        model=object(),
        feature_names="one_minus_iou, centroid_distance",
    )

    assert model.feature_names == ("one_minus_iou", "centroid_distance")


@pytest.mark.parametrize("feature_names", ["", ("one_minus_iou", True), ()])
def test_calibrated_association_model_rejects_invalid_feature_names(feature_names):
    with pytest.raises(ValueError, match="feature_names"):
        CalibratedAssociationModel(model=object(), feature_names=feature_names)


def test_reference_training_options_normalize_direct_string_sequences():
    options = ReferenceTrainingOptions(
        feature_names="one_minus_iou, centroid_distance",
        auto_registration_candidates="affine, rigid",
    )

    assert options.feature_names == ("one_minus_iou", "centroid_distance")
    assert options.auto_registration_candidates == ("affine", "rigid")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("curated_only", "false"),
        ("weighted_centroids", 1),
        ("restrict_training_to_reference_rois", "true"),
        ("velocity_variance", True),
        ("velocity_variance", float("nan")),
        ("velocity_variance", float("inf")),
        ("velocity_variance", -1.0),
        ("regularization", True),
        ("regularization", float("nan")),
        ("regularization", float("inf")),
        ("regularization", -1.0e-6),
        ("feature_names", ""),
        ("feature_names", ("one_minus_iou", True)),
        ("feature_names", ()),
        ("auto_registration_candidates", ("affine", "")),
        ("pairwise_cost_kwargs", (("large_cost", 1.0),)),
        ("transform_type", ""),
        ("order", None),
        ("fov_affine_mask_warp_mode", ""),
    ],
)
def test_reference_training_options_reject_invalid_controls(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        ReferenceTrainingOptions(**{field_name: value})
