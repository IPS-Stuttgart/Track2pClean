from __future__ import annotations

import numpy as np

from bayescatrack.reference import score_pairwise_matches


def test_score_pairwise_matches_counts_duplicate_predictions_as_false_positives():
    scores = score_pairwise_matches(
        predicted_pairs=np.array([[1, 2], [1, 2]], dtype=int),
        reference_pairs=np.array([[1, 2]], dtype=int),
    )

    assert scores["true_positives"] == 1
    assert scores["false_positives"] == 1
    assert scores["false_negatives"] == 0
    assert scores["precision"] == 0.5
    assert scores["recall"] == 1.0
    assert scores["f1"] == 2.0 / 3.0


def test_score_pairwise_matches_counts_duplicate_references_as_false_negatives():
    scores = score_pairwise_matches(
        predicted_pairs=np.array([[1, 2]], dtype=int),
        reference_pairs=np.array([[1, 2], [1, 2]], dtype=int),
    )

    assert scores["true_positives"] == 1
    assert scores["false_positives"] == 0
    assert scores["false_negatives"] == 1
