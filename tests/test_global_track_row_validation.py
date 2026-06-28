from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.tracking import _coerce_global_track_rows


@pytest.mark.parametrize(
    "bad_value",
    [True, np.bool_(False), 1.25, np.float64(np.inf), "2", -2, np.int64(-3)],
)
def test_global_track_rows_reject_malformed_roi_entries(bad_value) -> None:
    track_rows = np.asarray([[0, bad_value]], dtype=object)

    with pytest.raises(ValueError, match="global assignment track matrix"):
        _coerce_global_track_rows(track_rows, fill_value=-1)


def test_global_track_rows_allow_configured_missing_and_integer_like_entries() -> None:
    track_rows = np.asarray([[0, None, -7, 2.0, np.int64(3)]], dtype=object)

    coerced = _coerce_global_track_rows(track_rows, fill_value=-7)

    np.testing.assert_array_equal(coerced, np.asarray([[0, -7, -7, 2, 3]], dtype=int))
