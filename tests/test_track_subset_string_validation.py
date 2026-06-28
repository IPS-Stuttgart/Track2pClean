from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.complete_track_scores import score_track_matrices


def test_score_track_matrices_rejects_bare_string_session_pairs():
    reference = np.array([[0, 1]], dtype=object)
    predicted = np.array([[0, 1]], dtype=object)

    with pytest.raises(ValueError, match="session_pairs must be"):
        score_track_matrices(predicted, reference, session_pairs="01")


def test_score_track_matrices_rejects_string_like_session_pair_entries():
    reference = np.array([[0, 1]], dtype=object)
    predicted = np.array([[0, 1]], dtype=object)

    with pytest.raises(ValueError, match="session_pairs entries must be"):
        score_track_matrices(predicted, reference, session_pairs=["01"])


def test_score_track_matrices_rejects_bare_string_complete_session_indices():
    reference = np.array([[0, 1]], dtype=object)
    predicted = np.array([[0, 1]], dtype=object)

    with pytest.raises(ValueError, match="complete_session_indices must be"):
        score_track_matrices(predicted, reference, complete_session_indices="01")


def test_score_track_matrices_still_accepts_string_indices_inside_sequences():
    reference = np.array([[0, 1]], dtype=object)
    predicted = np.array([[0, 1]], dtype=object)

    scores = score_track_matrices(
        predicted,
        reference,
        session_pairs=(("0", "1"),),
        complete_session_indices=("0", "1"),
    )

    assert scores["pairwise_true_positives"] == 1
    assert scores["complete_track_true_positives"] == 1
