from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.experiments._full_mht_manifest_integration import (
    _uses_calibrated_association,
)
from bayescatrack.experiments.full_mht_local_context_integration import (
    install_full_mht_local_context_likelihood_gate,
)

full_mht = pytest.importorskip(
    "bayescatrack.experiments.track2p_policy_full_mht_benchmark"
)


def test_manifest_integration_detects_calibrated_local_context_rows():
    assert _uses_calibrated_association(
        {"association_score_mode": "calibrated-likelihood"}
    )
    assert not _uses_calibrated_association({"association_score_mode": "heuristic"})
    assert not _uses_calibrated_association({})


def test_local_context_likelihood_gate_zeros_feature_when_disabled(monkeypatch):
    observed: dict[str, np.ndarray] = {}

    def fake_likelihood(*, local_deformation, config, **_kwargs):
        observed["local_deformation"] = np.asarray(local_deformation, dtype=float)
        return observed["local_deformation"]

    monkeypatch.setattr(
        full_mht,
        "_bayescatrack_full_mht_local_context_gate",
        False,
        raising=False,
    )
    monkeypatch.setattr(full_mht, "_association_log_likelihood_matrix", fake_likelihood)

    install_full_mht_local_context_likelihood_gate()
    output = full_mht._association_log_likelihood_matrix(
        local_deformation=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        config=SimpleNamespace(local_deformation_weight=0.0),
    )

    assert np.array_equal(output, np.zeros((2, 2)))
    assert np.array_equal(observed["local_deformation"], np.zeros((2, 2)))


def test_local_context_likelihood_gate_preserves_feature_when_enabled(monkeypatch):
    observed: dict[str, np.ndarray] = {}

    def fake_likelihood(*, local_deformation, config, **_kwargs):
        observed["local_deformation"] = np.asarray(local_deformation, dtype=float)
        return observed["local_deformation"]

    local = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(
        full_mht,
        "_bayescatrack_full_mht_local_context_gate",
        False,
        raising=False,
    )
    monkeypatch.setattr(full_mht, "_association_log_likelihood_matrix", fake_likelihood)

    install_full_mht_local_context_likelihood_gate()
    output = full_mht._association_log_likelihood_matrix(
        local_deformation=local,
        config=SimpleNamespace(local_deformation_weight=0.5),
    )

    assert np.array_equal(output, local)
    assert np.array_equal(observed["local_deformation"], local)


def test_calibrated_likelihood_flips_scan_assignment_from_local_overlap(monkeypatch):
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.95)
    likelihood_config = full_mht.FullMHTConfig(
        association_score_mode="calibrated-likelihood",
        association_likelihood_weight=1.0,
        association_likelihood_clip=4.0,
        local_deformation_weight=0.5,
        edge_top_k=3,
        scan_hypotheses=1,
        min_edge_score=-10.0,
    )
    registered = np.asarray(
        [
            [0.70, 0.95, 0.20],
            [0.10, 0.90, 0.20],
            [0.20, 0.10, 0.92],
        ],
        dtype=float,
    )
    shifted = np.asarray(
        [
            [0.70, 0.10, 0.20],
            [0.10, 0.90, 0.20],
            [0.20, 0.10, 0.92],
        ],
        dtype=float,
    )
    centroid = np.asarray(
        [
            [1.0, 1.0, 8.0],
            [8.0, 1.0, 8.0],
            [8.0, 8.0, 1.0],
        ],
        dtype=float,
    )
    area = np.asarray(
        [
            [0.95, 0.95, 0.40],
            [0.40, 0.96, 0.40],
            [0.40, 0.40, 0.97],
        ],
        dtype=float,
    )
    growth = np.asarray(
        [
            [0.20, 7.00, 5.00],
            [5.00, 0.20, 5.00],
            [5.00, 5.00, 0.20],
        ],
        dtype=float,
    )
    mahal = np.asarray(
        [
            [0.30, 8.00, 6.00],
            [6.00, 0.30, 6.00],
            [6.00, 6.00, 0.30],
        ],
        dtype=float,
    )
    local = np.asarray(
        [
            [0.02, 1.00, 0.70],
            [0.70, 0.02, 0.70],
            [0.70, 0.70, 0.02],
        ],
        dtype=float,
    )
    likelihood = full_mht._association_log_likelihood_matrix(
        registered_iou=registered,
        shifted_iou=shifted,
        centroid_distance=centroid,
        area_ratio=area,
        target_cell_probabilities=np.asarray([0.95, 0.95, 0.95], dtype=float),
        threshold=0.5,
        growth_residual=growth,
        growth_mahalanobis=mahal,
        local_deformation=local,
        config=likelihood_config,
    )
    matrices = full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5, 6, 7], dtype=int),
        target_indices=np.asarray([9, 10, 11], dtype=int),
        registered_iou=registered,
        shifted_iou=shifted,
        centroid_distance=centroid,
        area_ratio=area,
        threshold=0.5,
        growth_residual=growth,
        growth_mahalanobis=mahal,
        local_deformation=local,
        growth_anchor_count=3,
        growth_model_type="synthetic",
        association_log_likelihood=likelihood,
    )
    monkeypatch.setattr(full_mht, "_sparse_pair_matrices", lambda *args, **kwargs: matrices)

    def fake_murty(cost_matrix, **_kwargs):
        row = np.asarray(cost_matrix[0], dtype=float)
        selected = int(np.argmin(row))
        return [{"assignment": np.asarray([selected], dtype=int), "cost": float(row[selected])}]

    monkeypatch.setattr(full_mht, "murty_k_best_assignments", fake_murty)
    hypothesis = full_mht._MHTHypothesis(
        tracks=np.asarray([[5, -1]], dtype=int),
        score=0.0,
        history=tuple(),
    )
    local_overlap_config = full_mht.FullMHTConfig(
        association_score_mode="heuristic",
        registered_iou_weight=1.0,
        shifted_iou_weight=0.0,
        area_ratio_weight=0.0,
        cell_probability_weight=0.0,
        centroid_distance_weight=0.0,
        threshold_margin_weight=0.0,
        growth_residual_weight=0.0,
        growth_mahalanobis_weight=0.0,
        local_deformation_weight=0.0,
        edge_top_k=3,
        scan_hypotheses=1,
        min_edge_score=-10.0,
    )

    local_only = full_mht._expand_hypothesis_scan(
        hypothesis,
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=0,
        config=local_overlap_config,
        track2p_prior_edges=frozenset(),
    )
    calibrated = full_mht._expand_hypothesis_scan(
        hypothesis,
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=0,
        config=likelihood_config,
        track2p_prior_edges=frozenset(),
    )

    assert local_only[0].tracks.tolist() == [[5, 10]]
    assert calibrated[0].tracks.tolist() == [[5, 9]]
    assert likelihood[0, 0] > likelihood[0, 1]
