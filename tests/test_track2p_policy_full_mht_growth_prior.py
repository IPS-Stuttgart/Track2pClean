from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace


pytest.importorskip("pyrecest")

from bayescatrack.experiments import (  # noqa: E402
    track2p_policy_full_mht_benchmark as full_mht,
)


def test_full_mht_growth_prior_is_fit_from_mutual_label_free_anchors():
    source_centroids = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=float
    )
    target_centroids = source_centroids + np.asarray([2.0, 3.0], dtype=float)
    registered_iou = np.asarray(
        [[0.95, 0.10, 0.10], [0.10, 0.92, 0.10], [0.10, 0.10, 0.90]],
        dtype=float,
    )
    shifted_iou = np.asarray(
        [[0.80, 0.05, 0.05], [0.05, 0.78, 0.05], [0.05, 0.05, 0.76]],
        dtype=float,
    )
    target_cell_probabilities = np.asarray([0.90, 0.91, 0.92], dtype=float)

    residual, mahalanobis, prior = full_mht._growth_residual_matrices(
        source_centroids,
        target_centroids,
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        target_cell_probabilities=target_cell_probabilities,
        config=full_mht.FullMHTConfig(),
    )

    assert prior.anchor_count == 3
    assert prior.model_type == "affine"
    assert np.max(np.diag(residual)) < 1.0e-5
    assert np.max(np.diag(mahalanobis)) < 1.0e-5
    assert residual[0, 1] > 9.0


def test_full_mht_growth_prior_ignores_nonmutual_candidate_spikes():
    config = full_mht.FullMHTConfig(
        growth_anchor_min_registered_iou=0.50,
        growth_anchor_min_shifted_iou=0.25,
        growth_anchor_min_cell_probability=0.80,
    )
    registered_iou = np.asarray([[0.90, 0.89], [0.88, 0.20]], dtype=float)
    shifted_iou = np.asarray([[0.70, 0.69], [0.68, 0.10]], dtype=float)
    target_cell_probabilities = np.asarray([0.95, 0.95], dtype=float)

    anchors = full_mht._mutual_growth_anchor_pairs(
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        target_cell_probabilities=target_cell_probabilities,
        config=config,
    )

    assert anchors == ((0, 0),)


def test_full_mht_active_track_sources_keep_recent_misses_only():
    tracks = np.asarray([[5, -1, -1], [7, 8, -1], [9, -1, 10]], dtype=int)

    no_gap = full_mht._active_track_sources(tracks, session_index=1, max_gap=0)
    one_gap = full_mht._active_track_sources(tracks, session_index=1, max_gap=1)

    assert [
        (src.row_index, src.source_session, src.source_roi, src.gap_length)
        for src in no_gap
    ] == [(1, 1, 8, 0)]
    assert [
        (src.row_index, src.source_session, src.source_roi, src.gap_length)
        for src in one_gap
    ] == [(0, 0, 5, 1), (1, 1, 8, 0), (2, 0, 9, 1)]


def test_full_mht_scan_can_reactivate_a_recently_missed_track(monkeypatch):
    hypothesis = full_mht._MHTHypothesis(
        tracks=np.asarray([[5, -1, -1]], dtype=int), score=0.0, history=tuple()
    )
    matrices = full_mht._FullMHTPairMatrices(
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.90]], dtype=float),
        shifted_iou=np.asarray([[0.80]], dtype=float),
        centroid_distance=np.asarray([[1.0]], dtype=float),
        area_ratio=np.asarray([[1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[0.0]], dtype=float),
        growth_mahalanobis=np.asarray([[0.0]], dtype=float),
        growth_anchor_count=1,
        growth_model_type="translation_fallback",
    )

    monkeypatch.setattr(
        full_mht,
        "_sparse_pair_matrices",
        lambda *args, **kwargs: matrices,
    )
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        full_mht,
        "murty_k_best_assignments",
        lambda *args, **kwargs: [
            {"assignment": np.asarray([0], dtype=int), "cost": -1.0}
        ],
    )

    output = full_mht._expand_hypothesis_scan(
        hypothesis,
        sessions=(object(), object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=1,
        config=full_mht.FullMHTConfig(
            max_gap=1,
            gap_reactivation_cost=0.0,
            min_edge_score=0.0,
        ),
    )

    assert output[0].tracks.tolist() == [[5, -1, 9]]
    assert output[0].history[-1]["gap_active_tracks"] == 1
    assert output[0].history[-1]["gap_reactivated_tracks"] == 1
