from __future__ import annotations

import numpy as np
from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.association.monotone_ranker import MonotoneRankerOptions, fit_monotone_ranking_association_model_from_blocks


def test_single_monotone_feature_name_string_is_not_split() -> None:
    feature_names = ("centroid_distance", "one_minus_iou", "session_gap")
    features = np.zeros((2, 2, len(feature_names)), dtype=float)
    features[..., 0] = np.array([[0.0, 3.0], [4.0, 0.0]], dtype=float)
    features[..., 1] = np.array([[0.0, 0.9], [0.8, 0.0]], dtype=float)
    features[..., 2] = 1.0
    block = ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=np.eye(2, dtype=int),
        reference_roi_indices=np.array([0, 1], dtype=int),
        measurement_roi_indices=np.array([0, 1], dtype=int),
        feature_names=feature_names,
    )

    model = fit_monotone_ranking_association_model_from_blocks(
        [block],
        options=MonotoneRankerOptions(monotone_feature_names="centroid_distance", max_iter=1),
    )

    assert model.monotone_feature_names == ("centroid_distance",)
