from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.edge_ranking import rank_labeled_edges


def test_rank_labeled_edges_accepts_binary_label_entries():
    rows = rank_labeled_edges(
        np.array([[True, 0.0], [0, 1.0]], dtype=object),
        {"cost": np.zeros((2, 2), dtype=float)},
        reference_roi_indices=np.array([10, 11]),
        measurement_roi_indices=np.array([20, 21]),
    )

    assert [
        (row["reference_roi_index"], row["measurement_roi_index"]) for row in rows
    ] == [
        (10, 20),
        (11, 21),
    ]


@pytest.mark.parametrize(
    "labels",
    [
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[2]]),
        np.array([[-1]]),
        np.array([[0.5]]),
        np.array([["0"]], dtype=object),
    ],
)
def test_rank_labeled_edges_rejects_non_binary_label_entries(labels):
    with pytest.raises(ValueError, match="labels"):
        rank_labeled_edges(
            labels,
            {"cost": np.zeros((1, 1), dtype=float)},
            reference_roi_indices=np.array([10]),
            measurement_roi_indices=np.array([20]),
        )
