from __future__ import annotations

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
        np.array([2]),
        -2,
    ],
)
def test_global_track_row_coercion_rejects_malformed_solver_entries(bad_value):
    track_rows = np.asarray([[0, bad_value]], dtype=object)

    with pytest.raises(ValueError, match="global assignment track matrix"):
        tracking._coerce_global_track_rows(track_rows, fill_value=-1)


def test_global_track_row_coercion_preserves_missing_and_integer_entries():
    track_rows = np.asarray([[0, None, -1, np.int64(3), 4.0]], dtype=object)

    coerced = tracking._coerce_global_track_rows(track_rows, fill_value=-1)

    npt.assert_array_equal(coerced, np.asarray([[0, -1, -1, 3, 4]], dtype=int))
