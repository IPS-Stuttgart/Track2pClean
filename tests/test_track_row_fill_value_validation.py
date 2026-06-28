import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.matching import (
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
)


@pytest.mark.parametrize(
    "bad_fill_value",
    [
        True,
        np.bool_(False),
        0,
        1,
        0.5,
        -1.5,
        np.nan,
        np.inf,
        "-1",
    ],
)
def test_build_track_rows_from_matches_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=np.array([0, 2]),
            fill_value=bad_fill_value,
        )


def test_build_track_rows_from_matches_preserves_negative_missing_sentinel():
    rows = build_track_rows_from_matches(
        ("s1", "s2"),
        [np.array([[0, 1]], dtype=int)],
        start_roi_indices=np.array([0, 2]),
        fill_value=np.int64(-9),
    )

    npt.assert_array_equal(rows, np.array([[0, 1], [2, -9]]))


def test_build_track_rows_from_bundles_rejects_colliding_fill_value_before_assignment():
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        build_track_rows_from_bundles([], fill_value=0)
