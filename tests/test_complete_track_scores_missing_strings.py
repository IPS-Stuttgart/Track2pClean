from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.evaluation.complete_track_scores import score_track_matrices


def test_score_track_matrices_accepts_string_missing_observations():
    reference = np.array([[0, 10, 20], [1, 11, 21]], dtype=object)
    predicted = np.array([["0", "10", "20"], ["1", "nan", "null"]], dtype=object)

    scores = score_track_matrices(predicted, reference)

    assert scores["pairwise_true_positives"] == 2
    assert scores["pairwise_false_positives"] == 0
    assert scores["pairwise_false_negatives"] == 2
    assert scores["pairwise_precision"] == pytest.approx(1.0)
    assert scores["pairwise_recall"] == pytest.approx(0.5)
    assert scores["pairwise_f1"] == pytest.approx(2.0 / 3.0)
    assert scores["complete_track_true_positives"] == 1
    assert scores["complete_track_false_positives"] == 0
    assert scores["complete_track_false_negatives"] == 1
    assert scores["complete_track_f1"] == pytest.approx(2.0 / 3.0)


def test_score_track_matrices_rejects_text_roi_observations():
    reference = np.array([[1, 2]], dtype=object)
    predicted = np.array([["abc", 2]], dtype=object)

    with pytest.raises(
        ValueError, match="predicted_track_matrix contains non-integer ROI index"
    ):
        score_track_matrices(predicted, reference)
