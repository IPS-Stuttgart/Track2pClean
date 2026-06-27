from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.track2p_metrics import (
    normalize_track_matrix,
    score_track_matrix_against_reference,
)
from bayescatrack.reference import Track2pReference


def test_track2p_metric_normalization_rejects_fractional_roi_indices():
    with pytest.raises(ValueError, match="ROI observations must be integer-like"):
        normalize_track_matrix([[0, 1.5]])


def test_track2p_metric_normalization_accepts_missing_observations():
    normalized = normalize_track_matrix([[0, "nan", None, -1]])

    assert normalized.shape == (1, 4)
    assert normalized[0, 0] == 0
    assert all(normalized[0, index] < 0 for index in (1, 2, 3))


def test_reference_scoring_treats_vector_prediction_as_single_track_row():
    reference = Track2pReference(
        session_names=("session_a", "session_b"),
        suite2p_indices=np.array([[0, 1]], dtype=object),
    )

    scores = score_track_matrix_against_reference([0, 1], reference)

    assert scores["pairwise_true_positives"] == 1
    assert scores["pairwise_false_positives"] == 0
    assert scores["complete_track_true_positives"] == 1
    assert scores["complete_track_false_positives"] == 0
