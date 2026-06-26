from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation import score_track_matrices


def test_score_track_matrices_rejects_repeated_session_pairs():
    reference = np.array([[1, 2, 3]], dtype=object)
    predicted = np.array([[1, 2, 3]], dtype=object)

    with pytest.raises(ValueError):
        score_track_matrices(
            predicted,
            reference,
            session_pairs=((0, 1), (0, 1)),
        )


def test_score_track_matrices_rejects_repeated_complete_session_indices():
    reference = np.array([[1, 2, 3]], dtype=object)
    predicted = np.array([[1, 2, 3]], dtype=object)

    with pytest.raises(ValueError):
        score_track_matrices(
            predicted,
            reference,
            complete_session_indices=(0, 1, 1),
        )
