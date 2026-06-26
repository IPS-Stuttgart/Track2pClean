from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("pyrecest")

from bayescatrack.experiments import (  # noqa: E402
    full_mht_growth_history_prediction_integration as growth_history,
)
from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht  # noqa: E402
from bayescatrack.experiments.full_mht_scan_history_dynamics_integration import (  # noqa: E402
    ScanHistoryEdgeFeatures,
)


def _config(**kwargs):
    dynamic = {
        "growth_history_prediction_weight": kwargs.pop("growth_history_prediction_weight", 1.0),
        "growth_history_prediction_scale": kwargs.pop("growth_history_prediction_scale", 1.0),
        "growth_history_prediction_clip": kwargs.pop("growth_history_prediction_clip", 8.0),
        "growth_history_prediction_min_edges": kwargs.pop("growth_history_prediction_min_edges", 1),
    }
    config = full_mht.FullMHTConfig(**kwargs)
    for key, value in dynamic.items():
        object.__setattr__(config, key, value)
    return config


def _matrices() -> full_mht._FullMHTPairMatrices:
    return full_mht._FullMHTPairMatrices(
        source_session=1,
        target_session=2,
        source_indices=np.asarray([9], dtype=int),
        target_indices=np.asarray([10, 11], dtype=int),
        registered_iou=np.asarray([[0.55, 0.95]], dtype=float),
        shifted_iou=np.asarray([[0.50, 0.95]], dtype=float),
        centroid_distance=np.asarray([[1.0, 1.0]], dtype=float),
        area_ratio=np.asarray([[1.0, 1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[0.60, 8.00]], dtype=float),
        growth_mahalanobis=np.asarray([[0.70, 8.50]], dtype=float),
        local_deformation=np.asarray([[0.03, 1.00]], dtype=float),
        growth_anchor_count=2,
        growth_model_type="affine",
    )


def test_row_growth_history_prediction_penalizes_motion_break() -> None:
    previous = (
        ScanHistoryEdgeFeatures(
            edge=(0, 1, 5, 9),
            registered_iou=0.90,
            shifted_iou=0.88,
            growth_residual=0.50,
            growth_mahalanobis=0.60,
            local_deformation=0.03,
        ),
    )
    coherent = ScanHistoryEdgeFeatures(
        edge=(1, 2, 9, 10),
        registered_iou=0.84,
        shifted_iou=0.82,
        growth_residual=0.80,
        growth_mahalanobis=0.90,
        local_deformation=0.05,
    )
    motion_break = ScanHistoryEdgeFeatures(
        edge=(1, 2, 9, 11),
        registered_iou=0.35,
        shifted_iou=0.25,
        growth_residual=7.00,
        growth_mahalanobis=8.00,
        local_deformation=1.00,
    )

    assert growth_history.row_growth_history_prediction_penalty(previous, coherent) == 0.0
    assert growth_history.row_growth_history_prediction_penalty(previous, motion_break) > 10.0


def test_growth_history_prediction_flips_scan_assignment_to_coherent_history(monkeypatch) -> None:
    growth_history.install_full_mht_growth_history_prediction_scoring()
    matrices = _matrices()
    hypothesis = full_mht._MHTHypothesis(
        tracks=np.asarray([[5, 9, -1]], dtype=int),
        score=0.0,
        history=(
            {
                "selected_edge_summaries": "0:5->1:9|prior=0|score=2|risk=0|veto=not_prior_edge|reg=0.9|shift=0.88|growth=0.5|mahal=0.6|local=0.03|cell=0.9"
            },
        ),
    )

    monkeypatch.setattr(full_mht, "_sparse_pair_matrices", lambda *args, **kwargs: matrices)
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 1.0)

    def fake_murty(cost_matrix, **_kwargs):
        row = np.asarray(cost_matrix[0], dtype=float)
        selected = int(np.argmin(row))
        return [{"assignment": np.asarray([selected], dtype=int), "cost": float(row[selected])}]

    monkeypatch.setattr(full_mht, "murty_k_best_assignments", fake_murty)

    local_only = full_mht._expand_hypothesis_scan(
        hypothesis,
        sessions=(object(), object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=1,
        config=_config(
            growth_history_prediction_weight=0.0,
            growth_residual_weight=0.0,
            growth_mahalanobis_weight=0.0,
            local_deformation_weight=0.0,
            edge_top_k=2,
            min_edge_score=-10.0,
        ),
    )
    history_aware = full_mht._expand_hypothesis_scan(
        hypothesis,
        sessions=(object(), object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=1,
        config=_config(
            growth_history_prediction_weight=1.0,
            growth_residual_weight=0.0,
            growth_mahalanobis_weight=0.0,
            local_deformation_weight=0.0,
            edge_top_k=2,
            min_edge_score=-10.0,
        ),
    )

    assert local_only[0].tracks.tolist() == [[5, 9, 11]]
    assert history_aware[0].tracks.tolist() == [[5, 9, 10]]
    assert "growth_pred=" in history_aware[0].history[-1]["selected_edge_summaries"]


def test_growth_history_prediction_needs_previous_identity_edge() -> None:
    growth_history.install_full_mht_growth_history_prediction_scoring()
    matrices = _matrices()
    hypothesis = full_mht._MHTHypothesis(
        tracks=np.asarray([[9, -1, -1]], dtype=int),
        score=0.0,
        history=tuple(),
    )
    config = _config(growth_history_prediction_weight=1.0)

    growth_history._CONTEXT_STACK.append(
        growth_history.GrowthHistoryPredictionContext(hypothesis=hypothesis, session_index=1)
    )
    try:
        penalty = growth_history.growth_history_prediction_penalty_for_candidate(
            matrices,
            source_local=0,
            target_local=1,
            target_session=2,
            config=config,
        )
    finally:
        growth_history._CONTEXT_STACK.pop()

    assert penalty == 0.0
