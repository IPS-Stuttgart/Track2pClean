from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import multisession_tracking


@pytest.mark.parametrize(
    "n_sessions",
    [
        True,
        np.bool_(False),
        2.5,
        np.float64(np.nan),
        np.asarray(2),
    ],
)
def test_tracks_to_matrix_rejects_ambiguous_session_count(n_sessions):
    with pytest.raises(ValueError, match="n_sessions must be a non-negative integer"):
        multisession_tracking._tracks_to_matrix(({0: 0},), n_sessions)


def test_tracks_to_matrix_accepts_integer_like_session_count():
    track_matrix = multisession_tracking._tracks_to_matrix(
        ({0: 4, 2: 7},),
        np.int64(3),
    )

    assert track_matrix.tolist() == [[4, -1, 7]]
