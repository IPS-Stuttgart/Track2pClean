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


def test_full_mht_local_deformation_penalizes_neighbor_inconsistent_edges():
    source_centroids = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=float
    )
    target_centroids = source_centroids + np.asarray([2.0, 3.0], dtype=float)
    deformation = full_mht._local_deformation_matrix(
        source_centroids,
        target_centroids,
        anchor_pairs=((0, 0), (1, 1), (2, 2)),
        affine_xy=np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=float),
    )

    assert np.max(np.diag(deformation)) < 1.0e-9
    assert deformation[1, 2] > 1.0


def test_full_mht_growth_context_returns_local_deformation_matrix():
    source_centroids = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=float
    )
    target_centroids = source_centroids + np.asarray([2.0, 3.0], dtype=float)
    registered_iou = np.eye(3, dtype=float)
    shifted_iou = np.eye(3, dtype=float)
    target_cell_probabilities = np.ones(3, dtype=float)

    _residual, _mahalanobis, deformation, prior = full_mht._growth_context_matrices(
        source_centroids,
        target_centroids,
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        target_cell_probabilities=target_cell_probabilities,
        config=full_mht.FullMHTConfig(
            growth_anchor_min_registered_iou=0.5,
            growth_anchor_min_shifted_iou=0.5,
        ),
    )

    assert prior.anchor_count == 3
    assert deformation.shape == (3, 3)
    assert np.max(np.diag(deformation)) < 1.0e-5


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


def test_full_mht_track2p_output_seed_source_uses_cell_gated_prediction_rows(monkeypatch):
    reference_tracks = np.asarray([[1, 2], [3, 4]], dtype=int)
    track2p_tracks = np.asarray(
        [[9, 10], [5, -1], [-1, 6], [9, 11], [7, 12]], dtype=int
    )
    cell_probability = {5: 0.8, 7: 0.2, 9: 0.9}
    monkeypatch.setattr(
        full_mht,
        "_cell_probability",
        lambda _sessions, _session, roi: cell_probability[int(roi)],
    )

    assert full_mht._seed_rois(
        sessions=(object(),),
        reference_tracks=reference_tracks,
        seed_session=0,
        seed_source="track2p-output",
        cell_probability_threshold=0.5,
        track2p_tracks=track2p_tracks,
    ) == [5, 9]


def test_full_mht_track2p_prior_edges_can_reuse_prediction_table():
    track2p_tracks = np.asarray([[5, 9, -1], [6, 10, 12], [-1, 11, 13]], dtype=int)

    assert full_mht._track2p_prior_edges(
        subject_dir=object(),
        config=object(),
        enabled=True,
        track2p_tracks=track2p_tracks,
    ) == frozenset(
        {
            (0, 1, 5, 9),
            (0, 1, 6, 10),
            (1, 2, 10, 12),
            (1, 2, 11, 13),
        }
    )


def test_full_mht_prunes_low_support_output_tracks():
    tracks = np.asarray(
        [[5, -1, -1], [6, 7, -1], [8, 9, 10]],
        dtype=int,
    )

    assert full_mht._prune_output_tracks(
        tracks, min_observations=1
    ).tolist() == tracks.tolist()
    assert full_mht._prune_output_tracks(
        tracks, min_observations=2
    ).tolist() == [[6, 7, -1], [8, 9, 10]]
    assert full_mht._prune_output_tracks(
        tracks, min_observations=3
    ).tolist() == [[8, 9, 10]]


def test_full_mht_scan_can_reactivate_a_recently_missed_track(monkeypatch):
    hypothesis = full_mht._MHTHypothesis(
        tracks=np.asarray([[5, -1, -1]], dtype=int), score=0.0, history=tuple()
    )
    matrices = full_mht._FullMHTPairMatrices(
        source_session=1,
        target_session=2,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.90]], dtype=float),
        shifted_iou=np.asarray([[0.80]], dtype=float),
        centroid_distance=np.asarray([[1.0]], dtype=float),
        area_ratio=np.asarray([[1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[0.0]], dtype=float),
        growth_mahalanobis=np.asarray([[0.0]], dtype=float),
        local_deformation=np.asarray([[0.0]], dtype=float),
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


def test_full_mht_edge_score_rewards_track2p_prior_edges(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.50]], dtype=float),
        shifted_iou=np.asarray([[0.25]], dtype=float),
        centroid_distance=np.asarray([[1.0]], dtype=float),
        area_ratio=np.asarray([[1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[0.0]], dtype=float),
        growth_mahalanobis=np.asarray([[0.0]], dtype=float),
        local_deformation=np.asarray([[0.0]], dtype=float),
        growth_anchor_count=0,
        growth_model_type="identity_no_anchors",
    )
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 1.0)

    config = full_mht.FullMHTConfig(
        track2p_prior_weight=2.0,
        track2p_non_prior_penalty=3.0,
    )
    without_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset(),
    )
    with_other_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 5, 10)}),
    )
    with_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 5, 9)}),
    )

    assert with_prior == without_prior + 2.0
    assert with_other_prior == without_prior - 3.0


def test_full_mht_track2p_prior_edge_risk_penalizes_suspicious_prior_edge(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.30]], dtype=float),
        shifted_iou=np.asarray([[0.25]], dtype=float),
        centroid_distance=np.asarray([[1.0]], dtype=float),
        area_ratio=np.asarray([[1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[0.0]], dtype=float),
        growth_mahalanobis=np.asarray([[3.0]], dtype=float),
        local_deformation=np.asarray([[0.0]], dtype=float),
        growth_anchor_count=0,
        growth_model_type="identity_no_anchors",
    )
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 1.0)

    config = full_mht.FullMHTConfig(
        track2p_prior_weight=2.0,
        track2p_prior_risk_mahalanobis_weight=4.0,
        track2p_prior_risk_mahalanobis_offset=1.5,
        track2p_prior_risk_registered_iou_weight=5.0,
        track2p_prior_risk_registered_iou_floor=0.5,
    )
    risk = full_mht._track2p_prior_edge_risk(
        registered_iou=0.30,
        growth_mahalanobis=3.0,
        config=config,
    )
    without_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset(),
    )
    with_prior = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 5, 9)}),
    )

    assert risk == pytest.approx(7.0)
    assert with_prior == pytest.approx(without_prior + 2.0 - risk)


def test_full_mht_miss_cost_penalizes_missing_track2p_prior_successor():
    active = full_mht._ActiveTrackSource(
        row_index=0, source_session=1, source_roi=5, gap_length=0
    )
    config = full_mht.FullMHTConfig(
        miss_cost=2.0,
        track2p_prior_miss_penalty=3.0,
    )

    assert full_mht._miss_cost(
        active,
        target_session=2,
        track2p_prior_edges=frozenset({(1, 2, 5, 9)}),
        config=config,
    ) == 5.0
    assert full_mht._miss_cost(
        active,
        target_session=3,
        track2p_prior_edges=frozenset({(1, 2, 5, 9)}),
        config=config,
    ) == 2.0


def test_full_mht_selected_edge_summary_is_label_free(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=1,
        target_session=2,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.50]], dtype=float),
        shifted_iou=np.asarray([[0.25]], dtype=float),
        centroid_distance=np.asarray([[1.0]], dtype=float),
        area_ratio=np.asarray([[1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[2.5]], dtype=float),
        growth_mahalanobis=np.asarray([[20.0]], dtype=float),
        local_deformation=np.asarray([[0.2]], dtype=float),
        growth_anchor_count=0,
        growth_model_type="identity_no_anchors",
    )
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.8)

    edge = full_mht._selected_edge_summary(
        (object(), object(), object()),
        matrices,
        active_source=full_mht._ActiveTrackSource(
            row_index=0, source_session=1, source_roi=5, gap_length=0
        ),
        target_session=2,
        target_roi=9,
        config=full_mht.FullMHTConfig(track2p_prior_weight=2.0),
        track2p_prior_edges=frozenset({(1, 2, 5, 9)}),
    )

    assert edge["is_track2p_prior"] == 1
    summary = str(edge["summary"])
    assert "1:5->2:9" in summary
    assert "prior=1" in summary
    assert "reg=0.5" in summary
    assert "shift=0.25" in summary
    assert "growth=2.5" in summary
    assert "mahal=20" in summary
    assert "cell=0.8" in summary
    assert "gt" not in summary.lower()


def test_full_mht_proposal_targets_are_scoped_to_source_and_scan():
    edges = frozenset(
        {
            (0, 1, 5, 9),
            (0, 1, 6, 10),
            (1, 2, 5, 11),
            (0, 2, 5, 12),
        }
    )

    assert full_mht._proposal_target_rois(
        edges, source_session=0, target_session=1, source_rois=(5,)
    ) == (9,)
    assert full_mht._proposal_target_rois(
        edges, source_session=0, target_session=1, source_rois=(5, 6)
    ) == (9, 10)
