from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyrecest")

from bayescatrack.experiments import (  # noqa: E402
    track2p_policy_full_mht_benchmark as full_mht,
)


def test_sparse_pair_matrices_source_indices_match_filtered_rows(monkeypatch):
    source_mask = np.zeros((4, 4), dtype=bool)
    source_mask[1:3, 1:3] = True
    target_mask = np.zeros((4, 4), dtype=bool)
    target_mask[1:3, 1:3] = True
    sessions = (
        SimpleNamespace(
            roi_indices=np.asarray([10], dtype=int),
            plane_data=SimpleNamespace(roi_masks=np.asarray([source_mask])),
        ),
        SimpleNamespace(
            roi_indices=np.asarray([20], dtype=int),
            plane_data=SimpleNamespace(roi_masks=np.asarray([target_mask])),
        ),
    )

    def roi_indices(session):
        return np.asarray(session.roi_indices, dtype=int)

    def registered_pair(sessions_arg, feature_cache, *, source_session, target_session):
        del sessions_arg, feature_cache, source_session
        return SimpleNamespace(
            roi_masks=sessions[int(target_session)].plane_data.roi_masks
        )

    def cell_probability(sessions_arg, session_index, roi):
        del sessions_arg, session_index, roi
        return 1.0

    def diagnostic_matrices(reference_masks, moving_masks, *, distance_threshold):
        del distance_threshold
        shape = (int(reference_masks.shape[0]), int(moving_masks.shape[0]))
        return (
            np.full(shape, 0.75, dtype=float),
            np.full(shape, 1.0, dtype=float),
            np.ones(shape, dtype=float),
        )

    def shifted_iou(reference_masks, moving_masks, *, radius):
        del radius
        return {
            "shifted_iou": np.ones(
                (int(reference_masks.shape[0]), int(moving_masks.shape[0])),
                dtype=float,
            )
        }

    def growth_residuals(
        source_centroids,
        target_centroids,
        *,
        registered_iou,
        shifted_iou,
        target_cell_probabilities,
        config,
    ):
        del source_centroids, target_centroids, shifted_iou
        del target_cell_probabilities, config
        shape = np.asarray(registered_iou, dtype=float).shape
        return (
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=float),
            SimpleNamespace(anchor_count=0, model_type="test"),
        )

    monkeypatch.setattr(full_mht, "_roi_indices", roi_indices)
    monkeypatch.setattr(full_mht, "_registered_pair", registered_pair)
    monkeypatch.setattr(full_mht, "_cell_probability", cell_probability)
    monkeypatch.setattr(
        full_mht,
        "_sparse_cross_iou_diagnostic_matrices",
        diagnostic_matrices,
    )
    monkeypatch.setattr(full_mht.rank, "_pairwise_shifted_iou_from_support", shifted_iou)
    monkeypatch.setattr(full_mht.rank, "_assignment_threshold", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(full_mht, "_growth_residual_matrices", growth_residuals)

    feature_cache = SimpleNamespace(
        transform_type="affine",
        iou_distance_threshold=5.0,
        cell_probability_threshold=0.0,
        threshold_method="min",
    )

    matrices = full_mht._sparse_pair_matrices(
        sessions,
        feature_cache,
        source_session=0,
        target_session=1,
        source_rois=(99, 10),
        edge_top_k=4,
        config=full_mht.FullMHTConfig(),
    )

    assert matrices.registered_iou.shape == (1, 1)
    assert matrices.source_indices.tolist() == [10]

    cached_matrices = full_mht._sparse_pair_matrices(
        sessions,
        feature_cache,
        source_session=0,
        target_session=1,
        source_rois=(99, 10),
        edge_top_k=4,
        config=full_mht.FullMHTConfig(),
    )
    assert cached_matrices.source_indices.tolist() == [10]
