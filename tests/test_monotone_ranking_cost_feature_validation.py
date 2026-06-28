from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.association.monotone_ranking_costs import (
    MonotoneRankerOptions,
    fit_monotone_ranked_association_model,
)


def _block(feature_names: tuple[str, ...]) -> ReferencePairwiseExamples:
    planes = {
        "centroid_distance": np.array([[0.1, 2.0], [2.0, 0.1]], dtype=float),
        "one_minus_iou": np.array([[0.1, 0.8], [0.7, 0.1]], dtype=float),
        "area_ratio_cost": np.array([[0.0, 0.4], [0.5, 0.0]], dtype=float),
    }
    duplicate_centroid_plane = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=float)
    occurrence_count: dict[str, int] = {}
    feature_planes = []
    for name in feature_names:
        occurrence_count[name] = occurrence_count.get(name, 0) + 1
        if name == "centroid_distance" and occurrence_count[name] > 1:
            feature_planes.append(duplicate_centroid_plane)
        else:
            feature_planes.append(planes[name])
    features = np.stack(feature_planes, axis=-1)
    return ReferencePairwiseExamples(
        session_a=0,
        session_b=1,
        features=features,
        labels=np.eye(2, dtype=int),
        reference_roi_indices=np.arange(2),
        measurement_roi_indices=np.arange(2),
        feature_names=feature_names,
    )


def test_monotone_ranking_costs_reject_duplicate_block_feature_names() -> None:
    block = _block(("centroid_distance", "centroid_distance", "one_minus_iou"))

    with pytest.raises(ValueError, match="block feature_names must be unique"):
        fit_monotone_ranked_association_model(
            [block],
            options=MonotoneRankerOptions(max_iter=1),
        )


def test_monotone_ranking_costs_reject_duplicate_requested_feature_names() -> None:
    block = _block(("centroid_distance", "one_minus_iou"))

    with pytest.raises(ValueError, match="feature_names must be unique"):
        fit_monotone_ranked_association_model(
            [block],
            feature_names=("centroid_distance", "centroid_distance"),
            options=MonotoneRankerOptions(max_iter=1),
        )


def test_monotone_ranking_costs_reject_duplicate_hardness_feature_names() -> None:
    block = _block(("centroid_distance", "one_minus_iou"))
    options = MonotoneRankerOptions(
        hardness_feature_names=("centroid_distance", "centroid_distance"),
        max_iter=1,
    )

    with pytest.raises(ValueError, match="hardness_feature_names must be unique"):
        fit_monotone_ranked_association_model([block], options=options)
