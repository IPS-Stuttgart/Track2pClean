from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


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


def test_full_mht_edge_score_penalizes_switching_from_prior_successor(monkeypatch):
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
        track2p_non_prior_penalty=3.0,
        track2p_prior_switch_penalty=4.0,
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
    switched = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 5, 10)}),
    )
    no_successor = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 6, 10)}),
    )

    assert switched == pytest.approx(without_prior - 7.0)
    assert no_successor == pytest.approx(without_prior - 3.0)


def test_full_mht_edge_score_penalizes_no_prior_successor_continuation(monkeypatch):
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
        track2p_non_prior_penalty=3.0,
        track2p_prior_switch_penalty=4.0,
        track2p_no_prior_successor_penalty=5.0,
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
    no_successor = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 6, 10)}),
    )
    switched = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset({(0, 1, 5, 10)}),
    )

    assert no_successor == pytest.approx(without_prior - 8.0)
    assert switched == pytest.approx(without_prior - 7.0)


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

    scan_disabled = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=full_mht.FullMHTConfig(
            track2p_prior_weight=2.0,
            track2p_prior_risk_mahalanobis_weight=4.0,
            track2p_prior_risk_mahalanobis_offset=1.5,
            track2p_prior_risk_registered_iou_weight=5.0,
            track2p_prior_risk_registered_iou_floor=0.5,
            track2p_prior_risk_scan_weight=0.0,
        ),
        track2p_prior_edges=frozenset({(0, 1, 5, 9)}),
    )
    assert scan_disabled == pytest.approx(without_prior + 2.0)


def test_full_mht_prior_veto_risk_accepts_strict_label_free_pocket():
    config = full_mht.FullMHTConfig(track2p_prior_veto_penalty=11.0)
    prior_edges = frozenset({(0, 1, 3, 5), (1, 2, 5, 9)})
    reason = full_mht._track2p_prior_veto_reason(
        registered_iou=0.50,
        shifted_iou=0.70,
        growth_residual=2.5,
        growth_mahalanobis=20.0,
        cell_probability_a=0.6,
        cell_probability_b=0.6,
        row_rank=1,
        column_rank=1,
        edge=(1, 2, 5, 9),
        n_sessions=3,
        track2p_prior_edges=prior_edges,
        config=config,
    )
    rejected = full_mht._track2p_prior_veto_reason(
        registered_iou=0.50,
        shifted_iou=0.70,
        growth_residual=2.5,
        growth_mahalanobis=20.0,
        cell_probability_a=0.9,
        cell_probability_b=0.9,
        row_rank=1,
        column_rank=1,
        edge=(1, 2, 5, 9),
        n_sessions=3,
        track2p_prior_edges=prior_edges,
        config=config,
    )
    risk = full_mht._track2p_prior_edge_risk(
        registered_iou=0.50,
        shifted_iou=0.70,
        growth_residual=2.5,
        growth_mahalanobis=20.0,
        cell_probability_a=0.6,
        cell_probability_b=0.6,
        row_rank=1,
        column_rank=1,
        edge=(1, 2, 5, 9),
        n_sessions=3,
        track2p_prior_edges=prior_edges,
        config=config,
    )

    assert reason == "accepted"
    assert rejected == "min_cell_probability_above_gate"
    assert risk == pytest.approx(11.0)


def test_full_mht_edge_score_applies_prior_veto_penalty(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=1,
        target_session=2,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.50]], dtype=float),
        shifted_iou=np.asarray([[0.70]], dtype=float),
        centroid_distance=np.asarray([[1.0]], dtype=float),
        area_ratio=np.asarray([[1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[2.5]], dtype=float),
        growth_mahalanobis=np.asarray([[20.0]], dtype=float),
        local_deformation=np.asarray([[0.0]], dtype=float),
        growth_anchor_count=2,
        growth_model_type="affine",
    )
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.6)
    config = full_mht.FullMHTConfig(
        track2p_prior_weight=12.0,
        track2p_prior_veto_penalty=20.0,
    )
    prior_edges = frozenset({(0, 1, 3, 5), (1, 2, 5, 9)})

    without_prior = full_mht._edge_score(
        (object(), object(), object()),
        matrices,
        target_session=2,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=frozenset(),
    )
    with_prior = full_mht._edge_score(
        (object(), object(), object()),
        matrices,
        target_session=2,
        source_local=0,
        target_local=0,
        config=config,
        track2p_prior_edges=prior_edges,
    )

    assert with_prior == pytest.approx(without_prior + 12.0 - 20.0)


def test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns():
    forbidden = {
        "edge_status_against_gt",
        "pairwise_delta_if_removed",
        "complete_delta_if_removed",
        "manual_gt",
        "manual-gt",
        "GROUND_TRUTH",
    }
    scoring_functions = (
        full_mht._edge_score,
        full_mht._track2p_prior_edge_risk,
        full_mht._track2p_prior_veto_reason,
        full_mht._selected_edge_summary,
    )

    source = "\n".join(inspect.getsource(function) for function in scoring_functions)

    for token in forbidden:
        assert token not in source


def test_full_mht_calibrated_likelihood_rewards_anchor_like_edges():
    registered = np.asarray(
        [[0.92, 0.12, 0.10], [0.11, 0.89, 0.13], [0.14, 0.10, 0.87]],
        dtype=float,
    )
    shifted = np.asarray(
        [[0.82, 0.05, 0.05], [0.05, 0.79, 0.06], [0.05, 0.04, 0.77]],
        dtype=float,
    )
    area = np.asarray(
        [[0.95, 0.40, 0.35], [0.42, 0.93, 0.38], [0.36, 0.41, 0.91]],
        dtype=float,
    )
    distance = np.asarray(
        [[1.0, 11.0, 12.0], [10.0, 1.2, 11.0], [12.0, 10.0, 0.8]],
        dtype=float,
    )
    residual = np.asarray(
        [[0.1, 5.0, 6.0], [5.5, 0.2, 5.0], [6.0, 5.2, 0.2]],
        dtype=float,
    )
    mahal = np.asarray(
        [[0.1, 4.0, 4.5], [4.3, 0.2, 4.0], [4.5, 4.2, 0.2]],
        dtype=float,
    )
    local = np.asarray(
        [[0.01, 0.50, 0.60], [0.40, 0.02, 0.50], [0.50, 0.45, 0.01]],
        dtype=float,
    )

    likelihood = full_mht._association_log_likelihood_matrix(
        registered_iou=registered,
        shifted_iou=shifted,
        centroid_distance=distance,
        area_ratio=area,
        target_cell_probabilities=np.asarray([0.9, 0.9, 0.9], dtype=float),
        threshold=0.0,
        growth_residual=residual,
        growth_mahalanobis=mahal,
        local_deformation=local,
        config=full_mht.FullMHTConfig(),
    )

    diagonal = np.diag(likelihood)
    off_diagonal = likelihood[~np.eye(3, dtype=bool)]
    assert float(np.min(diagonal)) > float(np.max(off_diagonal))
    assert float(np.min(diagonal)) > 0.0


def test_full_mht_edge_score_can_use_calibrated_likelihood_matrix(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([9], dtype=int),
        registered_iou=np.asarray([[0.10]], dtype=float),
        shifted_iou=np.asarray([[0.05]], dtype=float),
        centroid_distance=np.asarray([[50.0]], dtype=float),
        area_ratio=np.asarray([[0.1]], dtype=float),
        threshold=0.0,
        growth_residual=np.asarray([[10.0]], dtype=float),
        growth_mahalanobis=np.asarray([[10.0]], dtype=float),
        local_deformation=np.asarray([[1.0]], dtype=float),
        growth_anchor_count=0,
        growth_model_type="identity_no_anchors",
        association_log_likelihood=np.asarray([[2.0]], dtype=float),
    )
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *args, **kwargs: 0.0)

    score = full_mht._edge_score(
        (object(), object()),
        matrices,
        target_session=1,
        source_local=0,
        target_local=0,
        config=full_mht.FullMHTConfig(
            association_score_mode="calibrated-likelihood",
            association_likelihood_weight=2.5,
        ),
        track2p_prior_edges=frozenset(),
    )

    assert score == pytest.approx(5.0)


def test_full_mht_terminal_history_risk_can_rerank_completed_hypotheses(monkeypatch):
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
        growth_mahalanobis=np.asarray([[3.0]], dtype=float),
        local_deformation=np.asarray([[0.0]], dtype=float),
        growth_anchor_count=0,
        growth_model_type="identity_no_anchors",
    )
    monkeypatch.setattr(
        full_mht,
        "_sparse_pair_matrices",
        lambda *args, **kwargs: matrices,
    )

    risky = full_mht._MHTHypothesis(
        np.asarray([[5, 9]], dtype=int), score=10.0, history=tuple()
    )
    safer = full_mht._MHTHypothesis(
        np.asarray([[5, 10]], dtype=int), score=9.0, history=tuple()
    )
    selected, metadata = full_mht._select_final_hypothesis(
        (risky, safer),
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        config=full_mht.FullMHTConfig(
            track2p_prior_risk_mahalanobis_weight=2.0,
            track2p_prior_risk_mahalanobis_offset=1.0,
            terminal_history_risk_weight=1.0,
        ),
        track2p_prior_edges=frozenset({(0, 1, 5, 9)}),
    )

    assert selected is safer
    assert metadata["terminal_selected_rank"] == 2
    assert metadata["terminal_history_risk"] == pytest.approx(0.0)
    assert metadata["terminal_adjusted_score"] == pytest.approx(9.0)


def test_full_mht_terminal_history_risk_uses_scan_history(monkeypatch):
    monkeypatch.setattr(
        full_mht,
        "_sparse_pair_matrices",
        lambda *args, **kwargs: pytest.fail("unexpected matrix recomputation"),
    )
    risky = full_mht._MHTHypothesis(
        np.asarray([[5, 9]], dtype=int),
        score=10.0,
        history=({"selected_prior_risk": 2.5},),
    )
    safer = full_mht._MHTHypothesis(
        np.asarray([[5, 10]], dtype=int),
        score=8.0,
        history=({"selected_prior_risk": 0.0},),
    )

    selected, metadata = full_mht._select_final_hypothesis(
        (risky, safer),
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        config=full_mht.FullMHTConfig(terminal_history_risk_weight=1.0),
        track2p_prior_edges=frozenset({(0, 1, 5, 9)}),
    )

    assert selected is safer
    assert metadata["terminal_selected_rank"] == 2
    assert metadata["terminal_history_risk"] == pytest.approx(0.0)


def test_full_mht_terminal_identity_history_risk_can_rerank_hypotheses():
    risky = full_mht._MHTHypothesis(
        np.asarray([[5, 9]], dtype=int),
        score=10.0,
        history=(
            {
                "selected_non_prior_edges": 2,
                "no_prior_successor_continuations": 2,
                "selected_prior_risk": 0.0,
            },
        ),
    )
    safer = full_mht._MHTHypothesis(
        np.asarray([[5, 10]], dtype=int),
        score=8.5,
        history=(
            {
                "selected_non_prior_edges": 0,
                "no_prior_successor_continuations": 0,
                "selected_prior_risk": 0.0,
            },
        ),
    )

    selected, metadata = full_mht._select_final_hypothesis(
        (risky, safer),
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        config=full_mht.FullMHTConfig(
            terminal_no_prior_successor_history_weight=1.0,
        ),
        track2p_prior_edges=frozenset(),
    )

    assert selected is safer
    assert metadata["terminal_selected_rank"] == 2
    assert metadata["terminal_identity_history_risk"] == pytest.approx(0.0)
    assert metadata["terminal_adjusted_score"] == pytest.approx(8.5)


def test_full_mht_beam_pruning_score_uses_terminal_identity_risk():
    risky = full_mht._MHTHypothesis(
        np.asarray([[5, 9]], dtype=int),
        score=10.0,
        history=({"no_prior_successor_continuations": 3},),
    )
    safer = full_mht._MHTHypothesis(
        np.asarray([[5, 10]], dtype=int),
        score=8.0,
        history=({"no_prior_successor_continuations": 0},),
    )

    default = full_mht.FullMHTConfig()
    risk_aware = full_mht.FullMHTConfig(
        terminal_no_prior_successor_history_weight=1.0
    )

    assert full_mht._beam_pruning_score(risky, config=default) > (
        full_mht._beam_pruning_score(safer, config=default)
    )
    assert full_mht._beam_pruning_score(risky, config=risk_aware) < (
        full_mht._beam_pruning_score(safer, config=risk_aware)
    )


def test_full_mht_identity_diverse_beam_keeps_lower_risk_bucket():
    high_score_risky = full_mht._MHTHypothesis(
        np.asarray([[5, 9]], dtype=int),
        score=10.0,
        history=({"no_prior_successor_continuations": 3},),
    )
    next_score_risky = full_mht._MHTHypothesis(
        np.asarray([[5, 10]], dtype=int),
        score=9.0,
        history=({"no_prior_successor_continuations": 3},),
    )
    lower_score_clean = full_mht._MHTHypothesis(
        np.asarray([[5, 11]], dtype=int),
        score=8.0,
        history=({"no_prior_successor_continuations": 0},),
    )
    hypotheses = (high_score_risky, next_score_risky, lower_score_clean)

    default = full_mht._prune_beam(
        hypotheses,
        config=full_mht.FullMHTConfig(beam_width=2),
    )
    diverse = full_mht._prune_beam(
        hypotheses,
        config=full_mht.FullMHTConfig(beam_width=2, identity_diverse_beam=True),
    )

    assert default == [high_score_risky, next_score_risky]
    assert diverse == [high_score_risky, lower_score_clean]


def test_full_mht_identity_diverse_beam_uses_full_event_signature():
    high_score_switch = full_mht._MHTHypothesis(
        np.asarray([[5, 9]], dtype=int),
        score=10.0,
        history=(
            {
                "no_prior_successor_continuations": 0,
                "switched_prior_successors": 2,
                "missed_prior_successors": 0,
                "missed_tracks": 0,
            },
        ),
    )
    next_score_switch = full_mht._MHTHypothesis(
        np.asarray([[5, 10]], dtype=int),
        score=9.0,
        history=(
            {
                "no_prior_successor_continuations": 0,
                "switched_prior_successors": 2,
                "missed_prior_successors": 0,
                "missed_tracks": 0,
            },
        ),
    )
    lower_score_clean = full_mht._MHTHypothesis(
        np.asarray([[5, 11]], dtype=int),
        score=8.0,
        history=(
            {
                "no_prior_successor_continuations": 0,
                "switched_prior_successors": 0,
                "missed_prior_successors": 0,
                "missed_tracks": 0,
            },
        ),
    )
    hypotheses = (high_score_switch, next_score_switch, lower_score_clean)

    default = full_mht._prune_beam(
        hypotheses,
        config=full_mht.FullMHTConfig(beam_width=2),
    )
    diverse = full_mht._prune_beam(
        hypotheses,
        config=full_mht.FullMHTConfig(beam_width=2, identity_diverse_beam=True),
    )

    assert default == [high_score_switch, next_score_switch]
    assert diverse == [high_score_switch, lower_score_clean]
    assert full_mht._identity_diversity_bucket(high_score_switch) != (
        full_mht._identity_diversity_bucket(lower_score_clean)
    )


def test_full_mht_scan_assignment_compacts_murty_target_columns(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5], dtype=int),
        target_indices=np.asarray([10, 20, 30], dtype=int),
        registered_iou=np.asarray([[0.1, 0.9, 0.8]], dtype=float),
        shifted_iou=np.asarray([[0.1, 0.9, 0.8]], dtype=float),
        centroid_distance=np.asarray([[10.0, 1.0, 2.0]], dtype=float),
        area_ratio=np.asarray([[1.0, 1.0, 1.0]], dtype=float),
        threshold=0.0,
        growth_residual=np.zeros((1, 3), dtype=float),
        growth_mahalanobis=np.zeros((1, 3), dtype=float),
        local_deformation=np.zeros((1, 3), dtype=float),
        growth_anchor_count=1,
        growth_model_type="test",
    )
    captured: dict[str, object] = {}

    def fake_sparse_pair_matrices(*_args, **_kwargs):
        return matrices

    def fake_edge_score(_sessions, _matrices, *, target_local, **_kwargs):
        return {0: 0.1, 1: 2.0, 2: 1.5}[int(target_local)]

    def fake_murty(cost_matrix, **kwargs):
        captured["cost_shape"] = tuple(np.asarray(cost_matrix).shape)
        captured["cost_matrix"] = np.asarray(cost_matrix, dtype=float).copy()
        captured["col_non_assignment_costs"] = np.asarray(
            kwargs["col_non_assignment_costs"], dtype=float
        ).copy()
        return [{"assignment": np.asarray([0], dtype=int), "cost": float(cost_matrix[0, 0])}]

    monkeypatch.setattr(full_mht, "_sparse_pair_matrices", fake_sparse_pair_matrices)
    monkeypatch.setattr(full_mht, "_edge_score", fake_edge_score)
    monkeypatch.setattr(full_mht, "murty_k_best_assignments", fake_murty)
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *_args, **_kwargs: 0.9)

    output = full_mht._expand_hypothesis_scan(
        full_mht._MHTHypothesis(
            np.asarray([[5, -1]], dtype=int),
            score=0.0,
            history=(),
        ),
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=0,
        config=full_mht.FullMHTConfig(
            edge_top_k=1,
            min_edge_score=0.25,
            scan_hypotheses=1,
        ),
    )

    assert captured["cost_shape"] == (1, 1)
    assert np.asarray(captured["col_non_assignment_costs"]).shape == (1,)
    assert output[0].tracks[0, 1] == 20
    assert output[0].history[-1]["scan_candidates"] == 1



def test_full_mht_scan_assignment_decomposes_independent_components(monkeypatch):
    matrices = full_mht._FullMHTPairMatrices(
        source_session=0,
        target_session=1,
        source_indices=np.asarray([5, 6], dtype=int),
        target_indices=np.asarray([20, 40], dtype=int),
        registered_iou=np.asarray([[0.9, 0.1], [0.1, 0.8]], dtype=float),
        shifted_iou=np.asarray([[0.9, 0.1], [0.1, 0.8]], dtype=float),
        centroid_distance=np.asarray([[1.0, 8.0], [8.0, 1.5]], dtype=float),
        area_ratio=np.ones((2, 2), dtype=float),
        threshold=0.0,
        growth_residual=np.zeros((2, 2), dtype=float),
        growth_mahalanobis=np.zeros((2, 2), dtype=float),
        local_deformation=np.zeros((2, 2), dtype=float),
        growth_anchor_count=2,
        growth_model_type="test",
    )
    captured_shapes: list[tuple[int, int]] = []

    def fake_sparse_pair_matrices(*_args, **_kwargs):
        return matrices

    def fake_edge_score(_sessions, _matrices, *, source_local, target_local, **_kwargs):
        scores = {
            (0, 0): 2.0,
            (1, 1): 1.5,
        }
        return scores.get((int(source_local), int(target_local)), 0.0)

    def fake_murty(cost_matrix, **_kwargs):
        cost_matrix = np.asarray(cost_matrix, dtype=float)
        captured_shapes.append(tuple(cost_matrix.shape))
        return [{"assignment": np.asarray([0], dtype=int), "cost": float(cost_matrix[0, 0])}]

    monkeypatch.setattr(full_mht, "_sparse_pair_matrices", fake_sparse_pair_matrices)
    monkeypatch.setattr(full_mht, "_edge_score", fake_edge_score)
    monkeypatch.setattr(full_mht, "murty_k_best_assignments", fake_murty)
    monkeypatch.setattr(full_mht, "_cell_probability", lambda *_args, **_kwargs: 0.9)

    output = full_mht._expand_hypothesis_scan(
        full_mht._MHTHypothesis(
            np.asarray([[5, -1], [6, -1]], dtype=int),
            score=0.0,
            history=(),
        ),
        sessions=(object(), object()),
        feature_cache=SimpleNamespace(cell_probability_threshold=0.5),
        session_index=0,
        config=full_mht.FullMHTConfig(
            edge_top_k=1,
            min_edge_score=0.25,
            scan_hypotheses=1,
        ),
    )

    assert captured_shapes == [(1, 1), (1, 1)]
    np.testing.assert_array_equal(output[0].tracks, np.asarray([[5, 20], [6, 40]]))
    assert output[0].history[-1]["scan_candidates"] == 2
    assert output[0].history[-1]["scan_assignment_components"] == 2
    assert output[0].history[-1]["scan_assignment_decomposed"] == 1
    assert output[0].history[-1]["scan_assignment_solver_calls"] == 2
    assert output[0].history[-1]["scan_assignment_largest_component_rows"] == 1
    assert output[0].history[-1]["scan_assignment_largest_component_cols"] == 1

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
    assert edge["track2p_prior_risk"] == pytest.approx(0.0)
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
