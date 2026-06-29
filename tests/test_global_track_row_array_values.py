from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.tracking import _coerce_global_track_rows


@pytest.mark.parametrize(
    "bad_value",
    [
        np.asarray(2),
        np.asarray(-1),
        np.asarray([2]),
    ],
)
def test_global_track_rows_reject_array_wrapped_roi_entries(bad_value) -> None:
    track_rows = np.asarray([[0, bad_value]], dtype=object)

    with pytest.raises(ValueError, match="global assignment track matrix"):
        _coerce_global_track_rows(track_rows, fill_value=-1)


def test_global_track_rows_still_allow_plain_numpy_integer_scalars() -> None:
    track_rows = np.asarray([[0, np.int64(2)]], dtype=object)

    coerced = _coerce_global_track_rows(track_rows, fill_value=-1)

    np.testing.assert_array_equal(coerced, np.asarray([[0, 2]], dtype=int))
