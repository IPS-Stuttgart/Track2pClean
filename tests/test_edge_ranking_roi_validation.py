from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.edge_ranking import (
    missing_reference_edge_rows,
    rank_labeled_edges,
)


def test_rank_labeled_edges_rejects_boolean_roi_indices() -> None:
    with pytest.raises(ValueError, match="reference_roi_indices"):
        rank_labeled_edges(
            np.array([[1]], dtype=int),
            {"cost": np.array([[0.0]], dtype=float)},
            reference_roi_indices=np.array([True], dtype=object),
            measurement_roi_indices=np.array([2], dtype=int),
        )


def test_rank_labeled_edges_rejects_fractional_roi_indices() -> None:
    with pytest.raises(ValueError, match="measurement_roi_indices"):
        rank_labeled_edges(
            np.array([[1]], dtype=int),
            {"cost": np.array([[0.0]], dtype=float)},
            reference_roi_indices=np.array([1], dtype=int),
            measurement_roi_indices=np.array([2.5], dtype=object),
        )


def test_missing_reference_edge_rows_rejects_malformed_reference_match() -> None:
    with pytest.raises(ValueError, match="reference_matches"):
        missing_reference_edge_rows(
            [(1.5, 2)],
            reference_roi_indices=np.array([1], dtype=int),
            measurement_roi_indices=np.array([2], dtype=int),
            score_names=("cost",),
        )


def test_edge_ranking_roi_validation_preserves_integer_like_numpy_scalars() -> None:
    rows = rank_labeled_edges(
        np.array([[1]], dtype=int),
        {"cost": np.array([[0.0]], dtype=float)},
        reference_roi_indices=np.array([np.int64(7)], dtype=object),
        measurement_roi_indices=np.array([np.int32(8)], dtype=object),
    )

    assert rows[0]["reference_roi_index"] == 7
    assert rows[0]["measurement_roi_index"] == 8
