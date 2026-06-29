from __future__ import annotations

import bayescatrack  # noqa: F401  # ensure validation hooks are installed
import bayescatrack.tracking as tracking
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(False),
        1.5,
        np.nan,
        "2",
        np.asarray([1]),
        -1,
        np.int64(-3),
    ],
)
def test_global_assignment_track_rows_reject_malformed_roi_values(bad_value):
    with pytest.raises(ValueError, match="global assignment track matrix.*ROI indices"):
        tracking._coerce_global_track_rows(  # pylint: disable=protected-access
            np.asarray([[0, bad_value]], dtype=object),
            fill_value=-9,
        )


def test_global_assignment_track_rows_preserve_missing_and_integer_values():
    rows = tracking._coerce_global_track_rows(  # pylint: disable=protected-access
        np.asarray(
            [
                [None, 2],
                [-9, np.int64(4)],
                [5.0, 6],
            ],
            dtype=object,
        ),
        fill_value=-9,
    )

    npt.assert_array_equal(
        rows,
        np.asarray(
            [
                [-9, 2],
                [-9, 4],
                [5, 6],
            ],
            dtype=int,
        ),
    )


def test_global_assignment_track_rows_reject_invalid_fill_value():
    with pytest.raises(
        ValueError, match="fill_value must be a negative integer sentinel"
    ):
        tracking._coerce_global_track_rows(  # pylint: disable=protected-access
            np.asarray([[0, None]], dtype=object),
            fill_value=0,
        )
