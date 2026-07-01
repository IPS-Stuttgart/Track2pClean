from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.edge_ranking import rank_labeled_edges


def test_rank_labeled_edges_normalizes_encoded_score_names_before_direction_inference() -> (
    None
):
    rows = rank_labeled_edges(
        np.asarray([[1, 0]], dtype=int),
        {"iou".encode("utf-8"): np.asarray([[0.1, 0.9]], dtype=float)},
        reference_roi_indices=np.asarray([7], dtype=int),
        measurement_roi_indices=np.asarray([1, 2], dtype=int),
    )

    assert rows[0]["score_name"] == "iou"
    assert rows[0]["score_direction"] == "similarity"
    assert rows[0]["row_rank"] == 2
    assert rows[0]["row_margin"] == pytest.approx(-0.8)


def test_rank_labeled_edges_normalizes_encoded_score_direction_names() -> None:
    rows = rank_labeled_edges(
        np.asarray([[1, 0]], dtype=int),
        {"teacher_score": np.asarray([[0.1, 0.9]], dtype=float)},
        reference_roi_indices=np.asarray([7], dtype=int),
        measurement_roi_indices=np.asarray([1, 2], dtype=int),
        score_directions={"teacher_score".encode("utf-8"): "similarity"},
    )

    assert rows[0]["score_direction"] == "similarity"
    assert rows[0]["row_rank"] == 2


def test_rank_labeled_edges_rejects_duplicate_normalized_score_matrix_names() -> None:
    with pytest.raises(ValueError, match="score_matrices"):
        rank_labeled_edges(
            np.asarray([[1]], dtype=int),
            {
                "cost": np.asarray([[0.0]], dtype=float),
                "cost".encode("utf-8"): np.asarray([[1.0]], dtype=float),
            },
            reference_roi_indices=np.asarray([7], dtype=int),
            measurement_roi_indices=np.asarray([1], dtype=int),
        )
