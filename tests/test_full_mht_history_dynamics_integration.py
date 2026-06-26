from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.experiments import full_mht_history_dynamics_integration as dynamics
from bayescatrack.experiments.full_mht_history_dynamics_integration import (
    HistoryEdgeFeatures,
    row_motion_history_risk,
)


def _features(
    *,
    registered_iou: float = 0.80,
    shifted_iou: float = 0.70,
    growth_residual: float = 1.0,
    growth_mahalanobis: float = 1.0,
    local_deformation: float = 0.10,
) -> HistoryEdgeFeatures:
    return HistoryEdgeFeatures(
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        growth_residual=growth_residual,
        growth_mahalanobis=growth_mahalanobis,
        local_deformation=local_deformation,
    )


def test_row_motion_history_risk_ignores_coherent_history() -> None:
    risk = row_motion_history_risk(
        [
            _features(
                registered_iou=0.80,
                shifted_iou=0.70,
                growth_residual=1.0,
            ),
            _features(
                registered_iou=0.78,
                shifted_iou=0.68,
                growth_residual=1.2,
            ),
            _features(
                registered_iou=0.82,
                shifted_iou=0.71,
                growth_residual=0.9,
            ),
        ]
    )

    assert risk == pytest.approx(0.0)


def test_row_motion_history_risk_penalizes_within_track_outlier() -> None:
    risk = row_motion_history_risk(
        [
            _features(),
            _features(),
            _features(
                registered_iou=0.40,
                shifted_iou=0.30,
                growth_residual=5.0,
                growth_mahalanobis=5.0,
                local_deformation=1.00,
            ),
        ]
    )

    assert risk > 4.0


def test_row_motion_history_risk_penalizes_missing_edge_features() -> None:
    risk = row_motion_history_risk([_features(), _features()], missing_features=2)

    assert risk >= 2.0


def test_history_dynamics_terminal_selector_can_rerank(monkeypatch) -> None:
    pytest.importorskip("pyrecest")
    from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht

    dynamics.install_full_mht_history_dynamics_objective()

    high_score_bad_history = full_mht._MHTHypothesis(
        np.asarray([[1, 2, 3]], dtype=int),
        10.0,
        tuple(),
    )
    lower_score_good_history = full_mht._MHTHypothesis(
        np.asarray([[10, 20, 30]], dtype=int),
        9.0,
        tuple(),
    )

    def fake_motion_risk(hypothesis, **_kwargs):
        if int(np.asarray(hypothesis.tracks, dtype=int)[0, 0]) == 1:
            return 2.0
        return 0.0

    monkeypatch.setattr(dynamics, "terminal_motion_history_risk", fake_motion_risk)
    config = full_mht.FullMHTConfig()
    object.__setattr__(config, "terminal_motion_history_weight", 1.0)

    selected, summary = full_mht._select_final_hypothesis(
        [high_score_bad_history, lower_score_good_history],
        sessions=tuple(),
        feature_cache=SimpleNamespace(),
        config=config,
        track2p_prior_edges=frozenset(),
    )

    assert selected is lower_score_good_history
    assert summary["terminal_selected_rank"] == 2
    assert summary["terminal_motion_history_risk"] == 0.0
    assert summary["terminal_adjusted_score"] == pytest.approx(9.0)
