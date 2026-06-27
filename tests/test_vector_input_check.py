from __future__ import annotations

import pytest

from bayescatrack.evaluation import score_track_matrix_against_reference
from bayescatrack.evaluation import score_track_matrices as package_score_track_matrices
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.reference import Track2pReference


def test_score_track_matrices_treats_vector_inputs_as_single_track_rows() -> None:
    scores = score_track_matrices([0, 10, 20], [0, 10, 20])

    assert scores["complete_tracks"] == 1
    assert scores["reference_complete_tracks"] == 1
    assert scores["complete_track_true_positives"] == 1
    assert scores["complete_track_f1"] == pytest.approx(1.0)
    assert scores["pairwise_links"] == 2
    assert scores["pairwise_true_positives"] == 2
    assert scores["pairwise_f1"] == pytest.approx(1.0)


def test_package_score_track_matrices_uses_vector_input_patch() -> None:
    scores = package_score_track_matrices([0, 10], [[0, 10]])

    assert scores["complete_tracks"] == 1
    assert scores["reference_complete_tracks"] == 1
    assert scores["complete_track_f1"] == pytest.approx(1.0)


def test_reference_scoring_uses_vector_input_patch() -> None:
    reference = Track2pReference(
        session_names=("day0", "day1"),
        suite2p_indices=[[0, 10]],
    )

    scores = score_track_matrix_against_reference([0, 10], reference)

    assert scores["complete_tracks"] == 1
    assert scores["reference_complete_tracks"] == 1
    assert scores["complete_track_true_positives"] == 1
    assert scores["complete_track_f1"] == pytest.approx(1.0)
    assert scores["pairwise_links"] == 1
    assert scores["pairwise_true_positives"] == 1
    assert scores["pairwise_f1"] == pytest.approx(1.0)
