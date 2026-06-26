from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.experiments.full_mht_prior_survival_model import (
    FEATURE_NAMES,
    PriorEdgeSurvivalConfig,
    PriorEdgeSurvivalDiagnostics,
    calibrate_prior_edge_survival_model,
    prior_edge_feature_matrix,
    pseudo_hazard_background_mask,
    pseudo_survival_anchor_mask,
    score_prior_edge_survival,
)


def _anchor_edges() -> tuple[PriorEdgeSurvivalDiagnostics, ...]:
    return (
        PriorEdgeSurvivalDiagnostics(
            registered_iou=0.92,
            shifted_iou=0.82,
            growth_residual=0.20,
            growth_mahalanobis=0.30,
            min_cell_probability=0.93,
            area_ratio=0.98,
            local_deformation=0.02,
            row_rank=1,
            column_rank=1,
        ),
        PriorEdgeSurvivalDiagnostics(
            registered_iou=0.88,
            shifted_iou=0.78,
            growth_residual=0.40,
            growth_mahalanobis=0.60,
            min_cell_probability=0.89,
            area_ratio=0.95,
            local_deformation=0.04,
            row_rank=1,
            column_rank=1,
        ),
        PriorEdgeSurvivalDiagnostics(
            registered_iou=0.85,
            shifted_iou=0.75,
            growth_residual=0.60,
            growth_mahalanobis=0.90,
            min_cell_probability=0.86,
            area_ratio=0.92,
            local_deformation=0.05,
            row_rank=1,
            column_rank=1,
        ),
    )


def _hazard_edges() -> tuple[PriorEdgeSurvivalDiagnostics, ...]:
    return (
        PriorEdgeSurvivalDiagnostics(
            registered_iou=0.36,
            shifted_iou=0.76,
            growth_residual=2.90,
            growth_mahalanobis=2.70,
            min_cell_probability=0.62,
            area_ratio=0.86,
            local_deformation=0.25,
            row_rank=1,
            column_rank=1,
            terminal_edge=True,
            last_session_edge=True,
            complete_component=True,
        ),
        PriorEdgeSurvivalDiagnostics(
            registered_iou=0.42,
            shifted_iou=0.62,
            growth_residual=2.60,
            growth_mahalanobis=3.20,
            min_cell_probability=0.66,
            area_ratio=0.78,
            local_deformation=0.30,
            row_rank=1,
            column_rank=2,
            terminal_edge=True,
            last_session_edge=True,
            complete_component=True,
        ),
        PriorEdgeSurvivalDiagnostics(
            registered_iou=0.48,
            shifted_iou=0.50,
            growth_residual=2.30,
            growth_mahalanobis=2.80,
            min_cell_probability=0.69,
            area_ratio=0.72,
            local_deformation=0.35,
            row_rank=2,
            column_rank=1,
            terminal_edge=True,
            last_session_edge=True,
            complete_component=True,
        ),
    )


def test_prior_edge_survival_model_separates_anchor_and_hazard_edges() -> None:
    diagnostics = _anchor_edges() + _hazard_edges()

    model = calibrate_prior_edge_survival_model(diagnostics)
    scores = model.log_survival_ratio((diagnostics[0], diagnostics[-1]))

    assert model.enabled
    assert scores[0] > 0.0
    assert scores[1] < 0.0
    assert score_prior_edge_survival(diagnostics[0], model) == pytest.approx(
        scores[0]
    )


def test_prior_edge_survival_pseudo_masks_are_label_free() -> None:
    diagnostics = _anchor_edges() + _hazard_edges()

    anchors = pseudo_survival_anchor_mask(diagnostics)
    hazards = pseudo_hazard_background_mask(diagnostics)

    assert anchors.tolist() == [True, True, True, False, False, False]
    assert hazards.tolist() == [False, False, False, True, True, True]


def test_prior_edge_survival_model_falls_back_without_support() -> None:
    diagnostics = _anchor_edges()[:1] + _hazard_edges()[:1]

    model = calibrate_prior_edge_survival_model(diagnostics)

    assert not model.enabled
    assert np.allclose(model.log_survival_ratio(diagnostics), 0.0)


def test_prior_edge_feature_matrix_orients_risk_features_as_survival_evidence() -> None:
    anchor, hazard = _anchor_edges()[0], _hazard_edges()[0]
    features = prior_edge_feature_matrix((anchor, hazard))
    index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

    assert features[0, index["registered_iou"]] > features[1, index["registered_iou"]]
    assert features[0, index["negative_growth_residual"]] > features[
        1, index["negative_growth_residual"]
    ]
    assert features[0, index["negative_growth_mahalanobis"]] > features[
        1, index["negative_growth_mahalanobis"]
    ]
    assert features[0, index["negative_log_column_rank"]] >= features[
        1, index["negative_log_column_rank"]
    ]


def test_prior_edge_survival_model_clips_scores() -> None:
    diagnostics = _anchor_edges() + _hazard_edges()
    config = PriorEdgeSurvivalConfig(score_clip=0.5)

    model = calibrate_prior_edge_survival_model(diagnostics, config=config)
    scores = model.log_survival_ratio(diagnostics)

    assert np.max(np.abs(scores)) <= 0.5
