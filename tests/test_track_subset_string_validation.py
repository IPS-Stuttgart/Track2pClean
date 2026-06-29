from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.complete_track_scores import score_track_matrices


def _one_track_matrices() -> tuple[np.ndarray, np.ndarray]:
    predicted = np.array([[0, 1]], dtype=object)
    reference = np.array([[0, 1]], dtype=object)
    return predicted, reference


def test_score_track_matrices_rejects_bare_string_session_pairs():
    predicted, reference = _one_track_matrices()

    with pytest.raises(ValueError, match="session_pairs must be"):
        score_track_matrices(predicted, reference, session_pairs="01")


def test_score_track_matrices_rejects_string_like_session_pair_entries():
    predicted, reference = _one_track_matrices()

    with pytest.raises(ValueError, match="session_pairs entries must be"):
        score_track_matrices(predicted, reference, session_pairs=["01"])


@pytest.mark.parametrize("bad_session_pairs", [0, 1.5])
def test_score_track_matrices_rejects_non_iterable_session_pairs(bad_session_pairs):
    predicted, reference = _one_track_matrices()

    with pytest.raises(ValueError, match="session_pairs must be"):
        score_track_matrices(predicted, reference, session_pairs=bad_session_pairs)


@pytest.mark.parametrize("bad_session_pair", [0, 1.5, (0,), (0, 1, 2)])
def test_score_track_matrices_rejects_malformed_session_pair_entries(bad_session_pair):
    predicted, reference = _one_track_matrices()

    with pytest.raises(ValueError, match="session_pairs entries must be"):
        score_track_matrices(predicted, reference, session_pairs=[bad_session_pair])


def test_score_track_matrices_rejects_bare_string_complete_session_indices():
    predicted, reference = _one_track_matrices()

    with pytest.raises(ValueError, match="complete_session_indices must be"):
        score_track_matrices(predicted, reference, complete_session_indices="01")


@pytest.mark.parametrize("bad_session_indices", [0, 1.5])
def test_score_track_matrices_rejects_non_iterable_complete_session_indices(
    bad_session_indices,
):
    predicted, reference = _one_track_matrices()

    with pytest.raises(ValueError, match="complete_session_indices must be"):
        score_track_matrices(
            predicted,
            reference,
            complete_session_indices=bad_session_indices,
        )


def test_score_track_matrices_still_accepts_string_indices_inside_sequences():
    predicted, reference = _one_track_matrices()

    scores = score_track_matrices(
        predicted,
        reference,
        session_pairs=(("0", "1"),),
        complete_session_indices=("0", "1"),
    )

    assert scores["pairwise_true_positives"] == 1
    assert scores["complete_track_true_positives"] == 1
