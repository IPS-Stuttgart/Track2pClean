from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.reference import Track2pReference, score_complete_tracks_against_reference


def test_score_complete_tracks_against_reference_accepts_empty_prediction_sequence() -> None:
    reference = Track2pReference(
        session_names=("day0", "day1", "day2"),
        suite2p_indices=np.array(
            [
                [0, 10, 20],
                [1, 11, 21],
            ],
            dtype=object,
        ),
    )

    scores = score_complete_tracks_against_reference([], reference)

    assert scores["T_rc"] == 0
    assert scores["T_c"] == 0
    assert scores["T_gt"] == 2
    assert scores["ct"] == pytest.approx(0.0)
