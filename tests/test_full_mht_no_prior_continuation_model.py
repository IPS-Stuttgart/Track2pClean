from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.experiments.full_mht_no_prior_continuation_model import (
    FEATURE_NAMES,
    NoPriorContinuationConfig,
    NoPriorContinuationDiagnostics,
    calibrate_no_prior_continuation_model,
    no_prior_continuation_feature_matrix,
    pseudo_continuation_anchor_mask,
    pseudo_death_background_mask,
    score_no_prior_continuation,
)


def _continuation_edges() -> tuple[NoPriorContinuationDiagnostics, ...]:
    return (
        NoPriorContinuationDiagnostics(
            registered_iou=0.91,
            shifted_iou=0.80,
            growth_residual=0.20,
            growth_mahalanobis=0.30,
            min_cell_probability=0.92,
            area_ratio=0.96,
            centroid_distance=1.0,
            threshold_margin=0.40,
            local_deformation=0.02,
            row_rank=1,
            column_rank=1,
        ),
        NoPriorContinuationDiagnostics(
            registered_iou=0.87,
            shifted_iou=0.76,
            growth_residual=0.45,
            growth_mahalanobis=0.70,
            min_cell_probability=0.88,
            area_ratio=0.93,
            centroid_distance=1.4,
            threshold_margin=0.35,
            local_deformation=0.04,
            row_rank=1,
            column_rank=1,
        ),
    )


def _death_edges() -> tuple[NoPriorContinuationDiagnostics, ...]:
    return (
        NoPriorContinuationDiagnostics(
            registered_iou=0.30,
            shifted_iou=0.20,
            growth_residual=4.0,
            growth_mahalanobis=5.0,
            min_cell_probability=0.62,
            area_ratio=0.50,
            centroid_distance=12.0,
            threshold_margin=-0.20,
            local_deformation=0.55,
            row_rank=2,
            column_rank=3,
        ),
        NoPriorContinuationDiagnostics(
            registered_iou=0.40,
            shifted_iou=0.32,
            growth_residual=3.0,
            growth_mahalanobis=4.5,
            min_cell_probability=0.68,
            area_ratio=0.60,
            centroid_distance=10.0,
            threshold_margin=-0.10,
            local_deformation=0.42,
            row_rank=3,
            column_rank=2,
        ),
    )


def test_no_prior_continuation_model_separates_continuation_and_death_edges() -> None:
    diagnostics = _continuation_edges() + _death_edges()

    model = calibrate_no_prior_continuation_model(diagnostics)
    scores = model.log_continuation_ratio((diagnostics[0], diagnostics[-1]))

    assert model.enabled
    assert scores[0] > 0.0
    assert scores[1] < 0.0
    assert score_no_prior_continuation(diagnostics[0], model) == pytest.approx(
        scores[0]
    )


def test_no_prior_continuation_pseudo_masks_are_label_free() -> None:
    diagnostics = _continuation_edges() + _death_edges()

    anchors = pseudo_continuation_anchor_mask(diagnostics)
    background = pseudo_death_background_mask(diagnostics)

    assert anchors.tolist() == [True, True, False, False]
    assert background.tolist() == [False, False, True, True]


def test_no_prior_continuation_model_falls_back_without_support() -> None:
    diagnostics = _continuation_edges()[:1] + _death_edges()[:1]

    model = calibrate_no_prior_continuation_model(diagnostics)

    assert not model.enabled
    assert np.allclose(model.log_continuation_ratio(diagnostics), 0.0)


def test_no_prior_continuation_feature_matrix_orients_evidence() -> None:
    continuation, death = _continuation_edges()[0], _death_edges()[0]
    features = no_prior_continuation_feature_matrix((continuation, death))
    index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

    assert features[0, index["registered_iou"]] > features[1, index["registered_iou"]]
    assert features[0, index["threshold_margin"]] > features[1, index["threshold_margin"]]
    assert features[0, index["negative_centroid_distance"]] > features[
        1, index["negative_centroid_distance"]
    ]
    assert features[0, index["negative_growth_mahalanobis"]] > features[
        1, index["negative_growth_mahalanobis"]
    ]


def test_no_prior_continuation_model_clips_scores() -> None:
    diagnostics = _continuation_edges() + _death_edges()
    config = NoPriorContinuationConfig(score_clip=0.5)

    model = calibrate_no_prior_continuation_model(diagnostics, config=config)
    scores = model.log_continuation_ratio(diagnostics)

    assert np.max(np.abs(scores)) <= 0.5
