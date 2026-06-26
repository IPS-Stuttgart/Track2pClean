from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.association.monotone_ranker import (
    MonotoneRankerOptions,
    fit_monotone_ranking_association_model_from_blocks,
)


def _example_block(feature_names: tuple[str, ...]) -> ReferencePairwiseExamples:
    features = np.zeros((2, 2, len(feature_names)), dtype=float)
    if feature_names:
        features[..., 0] = np.array([[0.0, 3.0], [4.0, 0.0]], dtype=float)
    if len(feature_names) > 1:
        features[..., 1] = np.array([[0.0, 0.9], [0.8, 0.0]], dtype=float)
    if len(feature_names) > 2:
        features[..., 2:] = 1.0
    return ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=np.eye(2, dtype=int),
        reference_roi_indices=np.array([0, 1], dtype=int),
        measurement_roi_indices=np.array([0, 1], dtype=int),
        feature_names=feature_names,
    )


def test_monotone_ranker_rejects_duplicate_training_feature_names() -> None:
    block = _example_block(("centroid_distance", "centroid_distance", "session_gap"))

    with pytest.raises(ValueError, match="feature_names must be unique"):
        fit_monotone_ranking_association_model_from_blocks(
            [block],
            options=MonotoneRankerOptions(max_iter=1),
        )


def test_monotone_ranker_rejects_duplicate_requested_monotone_features() -> None:
    block = _example_block(("centroid_distance", "one_minus_iou", "session_gap"))
    options = MonotoneRankerOptions(
        monotone_feature_names=("centroid_distance", "centroid_distance"),
        max_iter=1,
    )

    with pytest.raises(ValueError, match="monotone_feature_names must be unique"):
        fit_monotone_ranking_association_model_from_blocks([block], options=options)
