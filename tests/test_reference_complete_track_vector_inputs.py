from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.reference import score_complete_tracks


def test_score_complete_tracks_treats_vector_inputs_as_single_track_rows():
    scores = score_complete_tracks([0, 10, 20], [0, 10, 20])

    assert scores["T_rc"] == 1
    assert scores["T_c"] == 1
    assert scores["T_gt"] == 1
    assert scores["ct"] == pytest.approx(1.0)


def test_score_complete_tracks_accepts_vector_against_explicit_row_matrix():
    scores = score_complete_tracks(
        np.asarray([0, 10, 20], dtype=object),
        np.asarray([[0, 10, 20]], dtype=object),
    )

    assert scores["perfectly_reconstructed_tracks"] == 1
    assert scores["reconstructed_complete_tracks"] == 1
    assert scores["ground_truth_complete_tracks"] == 1
    assert scores["complete_tracks_score"] == pytest.approx(1.0)


def test_score_complete_tracks_keeps_session_count_validation_for_vectors():
    with pytest.raises(ValueError, match="same number of sessions"):
        score_complete_tracks([0, 10], [[0, 10, 20]])
