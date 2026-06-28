import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.registration import RegisteredConsecutiveBundles
from bayescatrack.tracking import SubjectTrackingResult, _coerce_global_track_rows


@pytest.mark.parametrize(
    "bad_fill_value",
    [
        True,
        np.bool_(False),
        np.asarray(-1),
        0,
        1,
        0.5,
        -1.5,
        np.nan,
        np.inf,
        "-1",
    ],
)
def test_subject_tracking_result_rejects_malformed_or_colliding_fill_value(
    bad_fill_value,
):
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        SubjectTrackingResult(
            sessions=(),
            registered_bundles=RegisteredConsecutiveBundles(bundles=[]),
            match_results=(),
            session_names=("s1", "s2"),
            track_rows=np.asarray([[0, 1]], dtype=int),
            link_costs=np.zeros((1, 1), dtype=float),
            fill_value=bad_fill_value,
        )


def test_subject_tracking_result_preserves_negative_missing_sentinel():
    result = SubjectTrackingResult(
        sessions=(),
        registered_bundles=RegisteredConsecutiveBundles(bundles=[]),
        match_results=(),
        session_names=("s1", "s2"),
        track_rows=np.asarray([[0, -9]], dtype=int),
        link_costs=np.full((1, 1), np.nan, dtype=float),
        fill_value=np.int64(-9),
    )

    assert result.fill_value == -9
    npt.assert_array_equal(result.track_lengths(), np.asarray([1], dtype=int))


def test_global_track_row_coercion_rejects_colliding_fill_value():
    with pytest.raises(ValueError, match="fill_value must be a negative integer"):
        _coerce_global_track_rows(
            np.asarray([[None, 1]], dtype=object),
            fill_value=0,
        )
