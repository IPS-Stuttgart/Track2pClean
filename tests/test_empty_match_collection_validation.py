from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from bayescatrack.matching import build_track_rows_from_matches


@pytest.mark.parametrize(
    "empty_matches",
    [
        [],
        np.asarray([], dtype=int),
        np.empty((0, 2), dtype=int),
    ],
)
def test_build_track_rows_from_matches_accepts_empty_match_collections(empty_matches):
    rows = build_track_rows_from_matches(
        ("s1", "s2"),
        [empty_matches],
        start_roi_indices=np.asarray([0, 2], dtype=int),
    )

    npt.assert_array_equal(rows, np.asarray([[0, -1], [2, -1]], dtype=int))


def test_tuple_based_empty_match_sequences_still_work():
    rows = build_track_rows_from_matches(
        ("s1", "s2"),
        [((), ())],
        start_roi_indices=np.asarray([0], dtype=int),
    )

    npt.assert_array_equal(rows, np.asarray([[0, -1]], dtype=int))


def test_build_track_rows_from_matches_rejects_empty_wrong_width_pair_matrix():
    with pytest.raises(TypeError, match="unsupported match representation"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.empty((0, 3), dtype=int)],
            start_roi_indices=np.asarray([0], dtype=int),
        )
