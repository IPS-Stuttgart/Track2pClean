from __future__ import annotations

from bayescatrack.experiments.full_mht_history_consistency_model import (
    IdentityHistoryConsistencyConfig,
    identity_history_consistency_risk,
)


def _edge(
    *,
    registered_iou: float = 0.82,
    shifted_iou: float = 0.86,
    min_cell_probability: float = 0.9,
    growth_residual: float = 0.7,
    growth_mahalanobis: float = 0.8,
    local_deformation: float = 0.05,
) -> dict[str, float]:
    return {
        "registered_iou": registered_iou,
        "shifted_iou": shifted_iou,
        "min_cell_probability": min_cell_probability,
        "growth_residual": growth_residual,
        "growth_mahalanobis": growth_mahalanobis,
        "local_deformation": local_deformation,
    }


def test_identity_history_consistency_is_disabled_by_default() -> None:
    history = [_edge(), _edge(registered_iou=0.84, growth_residual=0.6)]
    candidate = _edge(
        registered_iou=0.35,
        shifted_iou=0.4,
        growth_residual=5.0,
        growth_mahalanobis=6.0,
    )

    assert identity_history_consistency_risk(history, candidate) == 0.0


def test_identity_history_consistency_ignores_short_histories() -> None:
    config = IdentityHistoryConsistencyConfig(weight=2.0, min_history_edges=2)
    candidate = _edge(
        registered_iou=0.35,
        shifted_iou=0.4,
        growth_residual=5.0,
        growth_mahalanobis=6.0,
    )

    assert identity_history_consistency_risk([_edge()], candidate, config=config) == 0.0


def test_identity_history_consistency_penalizes_joint_outlier() -> None:
    config = IdentityHistoryConsistencyConfig(
        weight=2.0,
        min_history_edges=2,
        min_feature_scale=0.05,
        joint_margin=0.5,
        score_clip=8.0,
    )
    history = [
        _edge(registered_iou=0.82, shifted_iou=0.84, growth_residual=0.6),
        _edge(registered_iou=0.86, shifted_iou=0.88, growth_residual=0.7),
        _edge(registered_iou=0.84, shifted_iou=0.86, growth_residual=0.5),
    ]
    candidate = _edge(
        registered_iou=0.42,
        shifted_iou=0.5,
        min_cell_probability=0.62,
        growth_residual=4.0,
        growth_mahalanobis=5.0,
        local_deformation=0.9,
    )

    assert identity_history_consistency_risk(history, candidate, config=config) > 0.0


def test_identity_history_consistency_requires_joint_evidence() -> None:
    config = IdentityHistoryConsistencyConfig(
        weight=3.0,
        min_history_edges=2,
        min_feature_scale=0.05,
        joint_margin=0.5,
    )
    history = [
        _edge(registered_iou=0.82, shifted_iou=0.84, growth_residual=0.6),
        _edge(registered_iou=0.86, shifted_iou=0.88, growth_residual=0.7),
        _edge(registered_iou=0.84, shifted_iou=0.86, growth_residual=0.5),
    ]
    low_overlap_only = _edge(
        registered_iou=0.42,
        shifted_iou=0.5,
        growth_residual=0.6,
        growth_mahalanobis=0.8,
    )
    high_growth_only = _edge(
        registered_iou=0.84,
        shifted_iou=0.86,
        growth_residual=4.0,
        growth_mahalanobis=5.0,
    )

    assert (
        identity_history_consistency_risk(history, low_overlap_only, config=config)
        == 0.0
    )
    assert (
        identity_history_consistency_risk(history, high_growth_only, config=config)
        == 0.0
    )


def test_identity_history_consistency_clips_unweighted_risk_before_weight() -> None:
    config = IdentityHistoryConsistencyConfig(
        weight=2.0,
        min_history_edges=2,
        min_feature_scale=0.01,
        joint_margin=0.0,
        score_clip=1.5,
    )
    history = [_edge(), _edge(), _edge()]
    candidate = _edge(
        registered_iou=0.0,
        shifted_iou=0.0,
        min_cell_probability=0.0,
        growth_residual=100.0,
        growth_mahalanobis=100.0,
        local_deformation=100.0,
    )

    assert identity_history_consistency_risk(history, candidate, config=config) == 3.0
