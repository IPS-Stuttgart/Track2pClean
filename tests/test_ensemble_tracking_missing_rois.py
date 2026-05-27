"""Regression tests for consensus ensembling with missing ROI sentinels."""

from __future__ import annotations

from collections import Counter

import numpy as np
from bayescatrack.association.ensemble_tracking import (
    consensus_track_rows,
    track_matrix_edge_counter,
)


def test_track_matrix_edge_counter_ignores_negative_missing_sentinel():
    matrix = np.array(
        [
            [0, 10, -1],
            [1, -1, 21],
            [-1, 12, 22],
        ],
        dtype=int,
    )

    assert track_matrix_edge_counter(matrix) == Counter(
        {
            (0, 1, 0, 10): 1,
            (1, 2, 12, 22): 1,
        }
    )


def test_consensus_track_rows_does_not_start_from_missing_negative_roi():
    matrices = [
        np.array([[0, 10], [-1, 11]], dtype=int),
        np.array([[0, 10], [-1, 12]], dtype=int),
    ]

    rows = consensus_track_rows(
        ["session0", "session1"],
        matrices,
        min_votes=1,
    )

    assert rows.tolist() == [[0, 10]]
